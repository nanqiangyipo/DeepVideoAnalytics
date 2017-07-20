from __future__ import absolute_import
import subprocess, sys, shutil, os, glob, time, logging, copy
import PIL
from django.conf import settings
from dva.celery import app
from .models import Video, Frame, TEvent, Query, IndexEntries, QueryResults, AppliedLabel, VDNDataset, Clusters, \
    ClusterCodes, Region, Tube, CustomDetector, Segment, IndexerQuery

from .operations.query_processing import IndexerTask,QueryProcessing
from .operations.detection_processing import DetectorTask
from .operations.video_processing import WFrame,WVideo
from dvalib import clustering

from collections import defaultdict
import calendar
import requests
import json
import zipfile
from . import serializers
import boto3
import random
from botocore.exceptions import ClientError
from .shared import handle_downloaded_file, create_video_folders, create_detector_folders, create_detector_dataset
from celery import group
from celery.result import allow_join_result


def get_queue_name(operation,args):
    if operation in settings.TASK_NAMES_TO_QUEUE:
        return settings.TASK_NAMES_TO_QUEUE[operation]
    elif 'index' in args:
        return settings.VISUAL_INDEXES[args['index']]['indexer_queue']
    else:
        raise NotImplementedError,"{}, {}".format(operation,args)


def perform_substitution(args,parent_task):
    """
    Its important to do a deep copy of args before executing any mutations.
    :param args:
    :param parent_task:
    :return:
    """
    args = copy.deepcopy(args) # IMPORTANT otherwise the first task to execute on the worker will fill the filters
    filters = args.get('filters',{})
    parent_args = json.loads(parent_task.arguments_json)
    if filters == '__parent__':
        parent_filters = parent_args.get('filters',{})
        logging.info('using filters from parent arguments: {}'.format(parent_args))
        args['filters'] = parent_filters
    elif filters:
        for k,v in args.get('filters',{}).items():
            if v == '__parent__':
                args['filters'][k] = parent_task.pk
            elif v == '__grand_parent__':
                args['filters'][k] = parent_task.parent.pk
    return args


def process_next(task_id):
    dt = TEvent.objects.get(pk=task_id)
    logging.info("next tasks for {}".format(dt.operation))
    for k in settings.POST_OPERATION_TASKS.get(dt.operation,[]):
        args = perform_substitution(k['arguments'], dt)
        jargs = json.dumps(args)
        logging.info("launching {}, {} with args {} as specified in config".format(dt.operation, k['task_name'], args))
        next_task = TEvent.objects.create(video=dt.video,operation=k['task_name'],arguments_json=jargs,parent=dt)
        app.send_task(k['task_name'], args=[next_task.pk, ], queue=get_queue_name(k['task_name'],args))
    for k in json.loads(dt.arguments_json).get('next_tasks',[]):
        args = perform_substitution(k['arguments'], dt)
        jargs = json.dumps(args)
        logging.info("launching {}, {} with args {} as specified in next_tasks".format(dt.operation, k['task_name'], args))
        next_task = TEvent.objects.create(video=dt.video,operation=k['task_name'], arguments_json=jargs,parent=dt)
        app.send_task(k['task_name'], args=[next_task.pk, ], queue=get_queue_name(k['task_name'],args))


def celery_40_bug_hack(start):
    """
    Celery 4.0.2 retries tasks due to ACK issues when running in solo mode,
    Since Tensorflow ncessiates use of solo mode, we can manually check if the task is has already run and quickly finis it
    Since the code never uses Celery results except for querying and retries are handled at application level this solves the
    issue
    :param start:
    :return:
    """
    return start.started


@app.task(track_started=True, name="perform_indexing", base=IndexerTask)
def perform_indexing(task_id):
    start = TEvent.objects.get(pk=task_id)
    if celery_40_bug_hack(start):
        return 0
    start.task_id = perform_indexing.request.id
    start.started = True
    start.operation = perform_indexing.name
    video_id = start.video_id
    dv = Video.objects.get(id=video_id)
    json_args = json.loads(start.arguments_json)
    target = json_args.get('target','frames')
    arguments = json_args.get('filters',{})
    index_name = json_args['index']
    start.save()
    start_time = time.time()
    video = WVideo(dv, settings.MEDIA_ROOT)
    arguments['video_id'] = dv.pk
    visual_index = perform_indexing.visual_indexer[index_name]
    if target == 'frames':
        frames = Frame.objects.all().filter(**arguments)
        index_name, index_results, feat_fname, entries_fname = video.index_frames(frames, visual_index,start.pk)
        detection_name = 'Frames_subset_by_{}'.format(start.pk)
        contains_frames = True
        contains_detections = False
    elif target == 'regions':
        detections = Region.objects.all().filter(**arguments)
        logging.info("Indexing {} Regions".format(detections.count()))
        detection_name = 'Faces_subset_by_{}'.format(start.pk) if index_name == 'facenet' else 'Regions_subset_by_{}'.format(start.pk)
        index_name, index_results, feat_fname, entries_fname = video.index_regions(detections, detection_name, visual_index)
        contains_frames = False
        contains_detections = True
    else:
        raise NotImplementedError
    i = IndexEntries()
    i.video = dv
    i.count = len(index_results)
    i.contains_detections = contains_detections
    i.contains_frames = contains_frames
    i.detection_name = detection_name
    i.algorithm = index_name
    i.entries_file_name = entries_fname.split('/')[-1]
    i.features_file_name = feat_fname.split('/')[-1]
    i.source = start
    i.source_filter_json = json.dumps(arguments)
    i.save()
    start.completed = True
    start.seconds = time.time() - start_time
    start.save()
    process_next(task_id)


@app.task(track_started=True, name="crop_regions_by_id")
def crop_regions_by_id(task_id):
    """
    Crop detected or annotated regions
    :param task_id:
    :return:
    """
    start = TEvent.objects.get(pk=task_id)
    if celery_40_bug_hack(start):
        return 0
    start.task_id = crop_regions_by_id.request.id
    start.started = True
    start.operation = crop_regions_by_id.name
    video_id = start.video_id
    kwargs = json.loads(start.arguments_json).get('filters',{})
    paths_to_regions = defaultdict(list)
    kwargs['video_id'] = start.video_id
    kwargs['materialized'] = False
    args = []
    queryset = Region.objects.all().filter(*args,**kwargs)
    for dr in queryset:
        path = "{}/{}/frames/{}.jpg".format(settings.MEDIA_ROOT,video_id,dr.parent_frame_index)
        paths_to_regions[path].append(dr)
    for path,regions in paths_to_regions.iteritems():
        img = PIL.Image.open(path)
        for dr in regions:
            img2 = img.crop((dr.x, dr.y,dr.x + dr.w, dr.y + dr.h))
            img2.save("{}/{}/regions/{}.jpg".format(settings.MEDIA_ROOT, video_id, dr.id))
            dr.materialized = True
    start.save()
    start_time = time.time()
    start.completed = True
    start.seconds = time.time() - start_time
    start.save()
    process_next(task_id)


@app.task(track_started=True, name="execute_index_subquery", base=IndexerTask)
def execute_index_subquery(query_id):
    iq = IndexerQuery.objects.get(id=query_id)
    start = TEvent()
    start.task_id = execute_index_subquery.request.id
    start.video_id = Video.objects.get(parent_query=iq.parent_query).pk
    start.started = True
    start.operation = execute_index_subquery.name
    start.save()
    qp = QueryProcessing()
    qp.load_from_db(iq.parent_query,settings.MEDIA_ROOT)
    # Its natural to question why "execute_index_subquery" is passed as an argument to the method below.
    # The reason behind this is to ensure that the network is loaded only once per SOLO celery worker process.
    # execute_index_subquery inherits IndexerTask which has "static" indexer objects. This ensures that
    # the network is only loaded once. A similar pattern can also be observed in inception_index_by_id .
    qp.execute_sub_query(iq,iq.algorithm,execute_index_subquery)
    start_time = time.time()
    start.completed = True
    start.seconds = time.time() - start_time
    start.save()
    return 0


@app.task(track_started=True, name="extract_frames")
def extract_frames(task_id):
    start = TEvent.objects.get(pk=task_id)
    if celery_40_bug_hack(start):
        return 0
    start.task_id = extract_frames.request.id
    start.started = True
    start.operation = extract_frames.name
    args = json.loads(start.arguments_json)
    if args == {}:
        args['rescale'] = 0
        args['rate'] = 30
        start.arguments_json = json.dumps(args)
    start.save()
    start_time = time.time()
    video_id = start.video_id
    dv = Video.objects.get(id=video_id)
    if dv.youtube_video:
        create_video_folders(dv)
    v = WVideo(dvideo=dv, media_dir=settings.MEDIA_ROOT)
    v.extract(args=args,start=start)
    process_next(task_id)
    start.completed = True
    start.seconds = time.time() - start_time
    start.save()
    os.remove("{}/{}/video/{}.zip".format(settings.MEDIA_ROOT, dv.pk, dv.pk))
    return 0


@app.task(track_started=True, name="segment_video")
def segment_video(task_id):
    start = TEvent.objects.get(pk=task_id)
    if celery_40_bug_hack(start):
        return 0
    start.task_id = segment_video.request.id
    start.started = True
    start.operation = segment_video.name
    args = json.loads(start.arguments_json)
    if 'rescale' not in args:
        args['rescale'] = 0
    if 'rate' not in args:
        args['rate'] = 30
    start.arguments_json = json.dumps(args)
    start.save()
    start_time = time.time()
    video_id = start.video_id
    dv = Video.objects.get(id=video_id)
    if dv.youtube_video:
        create_video_folders(dv)
    v = WVideo(dvideo=dv, media_dir=settings.MEDIA_ROOT)
    v.get_metadata()
    v.segment_video()
    decodes = []
    if args.get('sync',False):
        next_args = json.dumps({'rescale': args['rescale'], 'rate': args['rate']})
        next_task = TEvent.objects.create(video=dv, operation='decode_video', arguments_json=next_args, parent=start)
        decode_video(next_task.pk)  # decode it synchronously for testing in Travis
    else:
        for ds in Segment.objects.all().filter(video=dv):
            next_args = json.dumps({'rescale':args['rescale'],'rate':args['rate'],'filters':{'segment_index':ds.segment_index}})
            next_task = TEvent.objects.create(video=dv, operation='decode_video', arguments_json=next_args, parent=start)
            decodes.append(next_task.pk)
        result = group([decode_video.s(i).set(queue=settings.TASK_NAMES_TO_QUEUE['decode_video']) for i in decodes]).apply_async()
        with allow_join_result():
            result.join()
    process_next(task_id)
    start.completed = True
    start.seconds = time.time() - start_time
    start.save()
    return 0


@app.task(track_started=True,name="decode_video",ignore_result=False)
def decode_video(task_id):
    start = TEvent.objects.get(pk=task_id)
    if celery_40_bug_hack(start):
        return 0
    start.task_id = decode_video.request.id
    start.started = True
    start.operation = decode_video.name
    args = json.loads(start.arguments_json)
    start.save()
    start_time = time.time()
    video_id = start.video_id
    dv = Video.objects.get(id=video_id)
    kwargs = args.get('filters',{})
    kwargs['video_id'] = video_id
    v = WVideo(dvideo=dv, media_dir=settings.MEDIA_ROOT)
    for ds in Segment.objects.filter(**kwargs):
        v.decode_segment(ds=ds,denominator=args['rate'],rescale=args['rescale'])
    process_next(task_id)
start.completed = True
    start.seconds = time.time() - start_time
    start.save()
    return task_id


@app.task(track_started=True, name="assign_open_images_text_tags_by_id")
def assign_open_images_text_tags_by_id(task_id):
    start = TEvent.objects.get(pk=task_id)
    if celery_40_bug_hack(start):
        return 0
    start.task_id = assign_open_images_text_tags_by_id.request.id
    start.started = True
    start.operation = assign_open_images_text_tags_by_id.name
    start.save()
    start_time = time.time()
    video_id = start.video_id
    annotator_process = subprocess.Popen(['fab', 'assign_tags:{}'.format(video_id)],
                                         cwd=os.path.join(os.path.abspath(__file__).split('tasks.py')[0], '../'))
    annotator_process.wait()
    if annotator_process.returncode != 0:
        start.errored = True
        start.error_message = "assign_text_tags_by_id failed with return code {}".format(annotator_process.returncode)
        start.seconds = time.time() - start_time
        start.save()
        raise ValueError, start.error_message
    process_next(task_id)
    start.completed = True
    start.seconds = time.time() - start_time
    start.save()
    return 0


@app.task(track_started=True, name="perform_ssd_detection_by_id",base=DetectorTask)
def perform_ssd_detection_by_id(task_id):
    start = TEvent.objects.get(pk=task_id)
    if celery_40_bug_hack(start):
        return 0
    start.task_id = perform_ssd_detection_by_id.request.id
    start.started = True
    start.operation = perform_ssd_detection_by_id.name
    start.save()
    start_time = time.time()
    video_id = start.video_id
    detector = perform_ssd_detection_by_id.get_static_detectors['coco_mobilenet']
    args = json.loads(start.arguments_json)
    if detector.session is None:
        logging.info("loading detection model")
        detector.load()
    dv = Video.objects.get(id=video_id)
    if 'filters' in args:
        kwargs = args['filters']
        kwargs['video_id'] = video_id
        frames = Frame.objects.all().filter(**kwargs)
        logging.info("Performing SSD Using filters {}".format(kwargs))
    else:
        frames = Frame.objects.all().filter(video=dv)
    dd_list = []
    path_list = []
    for df in frames:
        local_path = "{}/{}/frames/{}.jpg".format(settings.MEDIA_ROOT,video_id,df.frame_index)
        detections = detector.detect(local_path)
        for d in detections:
            dd = Region()
            dd.region_type = Region.DETECTION
            dd.video_id = dv.pk
            dd.frame_id = df.pk
            dd.parent_frame_index = df.frame_index
            dd.parent_segment_index = df.segment_index
            dd.object_name = 'SSD_{}'.format(d['object_name'])
            dd.confidence = 100.0*d['score']
            dd.x = d['x']
            dd.y = d['y']
            dd.w = d['w']
            dd.h = d['h']
            dd.event_id = task_id
            dd_list.append(dd)
            path_list.append(local_path)
    dd_ids = Region.objects.bulk_create(dd_list,1000)
    process_next(task_id)
    start.completed = True
    start.seconds = time.time() - start_time
    start.save()
    return 0


@app.task(track_started=True, name="perform_textbox_detection_by_id")
def perform_textbox_detection_by_id(task_id):
    start = TEvent.objects.get(pk=task_id)
    if celery_40_bug_hack(start):
        return 0
    start.task_id = perform_textbox_detection_by_id.request.id
    start.started = True
    start.operation = perform_textbox_detection_by_id.name
    start.save()
    start_time = time.time()
    video_id = start.video_id
    detector = subprocess.Popen(['fab', 'detect_text_boxes:{}'.format(video_id)],
                                cwd=os.path.join(os.path.abspath(__file__).split('tasks.py')[0], '../'))
    detector.wait()
    if detector.returncode != 0:
        start.errored = True
        start.error_message = "fab detect_text_boxes failed with return code {}".format(detector.returncode)
        start.seconds = time.time() - start_time
        start.save()
        raise ValueError, start.error_message
    process_next(task_id)
    start.completed = True
    start.seconds = time.time() - start_time
    start.save()
    return 0


@app.task(track_started=True, name="perform_text_recognition_by_id")
def perform_text_recognition_by_id(task_id):
    start = TEvent.objects.get(pk=task_id)
    if celery_40_bug_hack(start):
        return 0
    start.task_id = perform_text_recognition_by_id.request.id
    start.started = True
    start.operation = perform_text_recognition_by_id.name
    start.save()
    start_time = time.time()
    video_id = start.video_id
    detector = subprocess.Popen(['fab', 'recognize_text:{}'.format(video_id)],
                                cwd=os.path.join(os.path.abspath(__file__).split('tasks.py')[0], '../'))
    detector.wait()
    if detector.returncode != 0:
        start.errored = True
        start.error_message = "fab recognize_text failed with return code {}".format(detector.returncode)
        start.seconds = time.time() - start_time
        start.save()
        raise ValueError, start.error_message
    process_next(task_id)
    start.completed = True
    start.seconds = time.time() - start_time
    start.save()
    return 0


@app.task(track_started=True, name="perform_face_detection",base=DetectorTask)
def perform_face_detection(task_id):
    start = TEvent.objects.get(pk=task_id)
    if celery_40_bug_hack(start):
        return 0
    start.task_id = perform_face_detection.request.id
    start.started = True
    start.operation = perform_face_detection.name
    start.save()
    start_time = time.time()
    video_id = start.video_id
    detector = perform_face_detection.get_static_detectors['face_mtcnn']
    if detector.session is None:
        logging.info("loading detection model")
        detector.load()
    input_paths = {}
    args = json.loads(start.arguments_json)
    filters_kwargs = args.get('filters',{})
    filters_kwargs['video_id'] = video_id
    for df in Frame.objects.all().filter(**filters_kwargs):
        input_paths["{}/{}/frames/{}.jpg".format(settings.MEDIA_ROOT,video_id,df.frame_index)] = df.pk
    faces_dir = '{}/{}/regions'.format(settings.MEDIA_ROOT, video_id)
    aligned_paths = {}
    for image_path in input_paths:
        aligned_paths[image_path] = detector.detect(image_path)
    logging.info(len(aligned_paths))
    faces = []
    faces_to_pk = {}
    for path, vlist in aligned_paths.iteritems():
        for v in vlist:
            d = Region()
            d.region_type = Region.DETECTION
            d.video_id = video_id
            d.confidence = 100.0
            d.frame_id = input_paths[path]
            d.object_name = "MTCNN_face"
            d.materialized = True
            d.event = start
            d.y = v['y']
            d.x = v['x']
            d.w = v['w']
            d.h = v['h']
            d.save()
            face_path = '{}/{}.jpg'.format(faces_dir, d.pk)
            output_filename = os.path.join(faces_dir, face_path)
            im = PIL.Image.fromarray(v['scaled'])
            im.save(output_filename)
            faces.append(face_path)
            faces_to_pk[face_path] = d.pk
    process_next(task_id)
    start.completed = True
    start.seconds = time.time() - start_time
    start.save()
    return 0


@app.task(track_started=True, name="export_video_by_id")
def export_video_by_id(task_id):
    start = TEvent.objects.get(pk=task_id)
    if celery_40_bug_hack(start):
        return 0
    start.task_id = export_video_by_id.request.id
    start.started = True
    start.operation = export_video_by_id.name
    start.save()
    start_time = time.time()
    video_id = start.video_id
    video_obj = Video.objects.get(pk=video_id)
    file_name = '{}_{}.dva_export.zip'.format(video_id, int(calendar.timegm(time.gmtime())))
    start.file_name = file_name
    try:
        os.mkdir("{}/{}".format(settings.MEDIA_ROOT, 'exports'))
    except:
        pass
    outdirname = "{}/exports/{}".format(settings.MEDIA_ROOT, video_id)
    if os.path.isdir(outdirname):
        shutil.rmtree(outdirname)
    shutil.copytree('{}/{}'.format(settings.MEDIA_ROOT, video_id),
                    "{}/exports/{}".format(settings.MEDIA_ROOT, video_id))
    a = serializers.VideoExportSerializer(instance=video_obj)
    with file("{}/exports/{}/table_data.json".format(settings.MEDIA_ROOT, video_id), 'w') as output:
        json.dump(a.data, output)
    zipper = subprocess.Popen(['zip', file_name, '-r', '{}'.format(video_id)],
                              cwd='{}/exports/'.format(settings.MEDIA_ROOT))
    zipper.wait()
    if zipper.returncode != 0:
        start.errored = True
        start.error_message = "Could not zip {}".format(zipper.returncode)
        start.seconds = time.time() - start_time
        start.save()
        raise ValueError, start.error_message
    shutil.rmtree("{}/exports/{}".format(settings.MEDIA_ROOT, video_id))
    start.completed = True
    start.seconds = time.time() - start_time
    start.save()
    return start.file_name


@app.task(track_started=True, name="import_video_by_id")
def import_video_by_id(task_id):
    start = TEvent.objects.get(pk=task_id)
    if celery_40_bug_hack(start):
        return 0
    start.task_id = import_video_by_id.request.id
    start.started = True
    start.operation = import_video_by_id.name
    start.save()
    start_time = time.time()
    video_id = start.video_id
    video_obj = Video.objects.get(pk=video_id)
    if video_obj.vdn_dataset and not video_obj.uploaded:
        if video_obj.vdn_dataset.aws_requester_pays:
            s3import = TEvent()
            s3import.video = video_obj
            s3import.key = video_obj.vdn_dataset.aws_key
            s3import.bucket = video_obj.vdn_dataset.aws_bucket
            s3import.requester_pays = True
            s3import.operation = "import_video_from_s3"
            s3import.save()
            app.send_task(s3import.operation, args=[s3import.pk, ],
                          queue=settings.TASK_NAMES_TO_QUEUE[s3import.operation])
            start.completed = True
            start.seconds = time.time() - start_time
            start.save()
            return 0
    zipf = zipfile.ZipFile("{}/{}/{}.zip".format(settings.MEDIA_ROOT, video_id, video_id), 'r')
    zipf.extractall("{}/{}/".format(settings.MEDIA_ROOT, video_id))
    zipf.close()
    video_root_dir = "{}/{}/".format(settings.MEDIA_ROOT, video_id)
    old_key = None
    for k in os.listdir(video_root_dir):
        unzipped_dir = "{}{}".format(video_root_dir, k)
        if os.path.isdir(unzipped_dir):
            for subdir in os.listdir(unzipped_dir):
                shutil.move("{}/{}".format(unzipped_dir, subdir), "{}".format(video_root_dir))
            shutil.rmtree(unzipped_dir)
            break
    with open("{}/{}/table_data.json".format(settings.MEDIA_ROOT, video_id)) as input_json:
        video_json = json.load(input_json)
    serializers.import_video_json(video_obj, video_json, video_root_dir)
    source_zip = "{}/{}.zip".format(video_root_dir, video_obj.pk)
    os.remove(source_zip)
    start.completed = True
    start.seconds = time.time() - start_time
    start.save()


@app.task(track_started=True, name="import_vdn_file")
def import_vdn_file(task_id):
    start = TEvent.objects.get(pk=task_id)
    if celery_40_bug_hack(start):
        return 0
    start.started = True
    start.task_id = import_vdn_file.request.id
    start.operation = import_vdn_file.name
    start.save()
    start_time = time.time()
    dv = start.video
    create_video_folders(dv, create_subdirs=False)
    if 'www.dropbox.com' in dv.vdn_dataset.download_url and not dv.vdn_dataset.download_url.endswith('?dl=1'):
        r = requests.get(dv.vdn_dataset.download_url + '?dl=1')
    else:
        r = requests.get(dv.vdn_dataset.download_url)
    output_filename = "{}/{}/{}.zip".format(settings.MEDIA_ROOT, dv.pk, dv.pk)
    with open(output_filename, 'wb') as f:
        for chunk in r.iter_content(chunk_size=1024):
            if chunk:
                f.write(chunk)
    r.close()
    zipf = zipfile.ZipFile("{}/{}/{}.zip".format(settings.MEDIA_ROOT, dv.pk, dv.pk), 'r')
    zipf.extractall("{}/{}/".format(settings.MEDIA_ROOT, dv.pk))
    zipf.close()
    video_root_dir = "{}/{}/".format(settings.MEDIA_ROOT, dv.pk)
    for k in os.listdir(video_root_dir):
        unzipped_dir = "{}{}".format(video_root_dir, k)
        if os.path.isdir(unzipped_dir):
            for subdir in os.listdir(unzipped_dir):
                shutil.move("{}/{}".format(unzipped_dir, subdir), "{}".format(video_root_dir))
            shutil.rmtree(unzipped_dir)
            break
    with open("{}/{}/table_data.json".format(settings.MEDIA_ROOT, dv.pk)) as input_json:
        video_json = json.load(input_json)
    serializers.import_video_json(dv, video_json, video_root_dir)
    source_zip = "{}/{}.zip".format(video_root_dir, dv.pk)
    os.remove(source_zip)
    dv.uploaded = True
    dv.save()
    process_next(task_id)
    start.completed = True
    start.seconds = time.time() - start_time
    start.save()


@app.task(track_started=True, name="import_vdn_detector_file")
def import_vdn_detector_file(task_id):
    start = TEvent.objects.get(pk=task_id)
    if celery_40_bug_hack(start):
        return 0
    start.started = True
    start.task_id = import_vdn_detector_file.request.id
    start.operation = import_vdn_detector_file.name
    start.save()
    start_time = time.time()
    dd = CustomDetector.objects.get(pk=json.loads(start.arguments_json)['detector_pk'])
    create_detector_folders(dd, create_subdirs=False)
    if 'www.dropbox.com' in dd.vdn_detector.download_url and not dd.vdn_detector.download_url.endswith('?dl=1'):
        r = requests.get(dd.vdn_detector.download_url + '?dl=1')
    else:
        r = requests.get(dd.vdn_detector.download_url)
    output_filename = "{}/detectors/{}.zip".format(settings.MEDIA_ROOT, dd.pk)
    with open(output_filename, 'wb') as f:
        for chunk in r.iter_content(chunk_size=1024):
            if chunk:
                f.write(chunk)
    r.close()
    source_zip = "{}/detectors/{}.zip".format(settings.MEDIA_ROOT, dd.pk)
    zipf = zipfile.ZipFile(source_zip, 'r')
    zipf.extractall("{}/detectors/{}/".format(settings.MEDIA_ROOT, dd.pk))
    zipf.close()
    serializers.import_detector(dd)
    dd.save()
    os.remove(source_zip)
    process_next(task_id)
    start.completed = True
    start.seconds = time.time() - start_time
    start.save()


@app.task(track_started=True, name="import_vdn_s3")
def import_vdn_s3(task_id):
    start = TEvent.objects.get(pk=task_id)
    if celery_40_bug_hack(start):
        return 0
    start.started = True
    start.task_id = import_vdn_s3.request.id
    start.operation = import_vdn_s3.name
    start.save()
    start_time = time.time()
    dv = start.video
    create_video_folders(dv, create_subdirs=False)
    client = boto3.client('s3')
    resource = boto3.resource('s3')
    key = dv.vdn_dataset.aws_key
    bucket = dv.vdn_dataset.aws_bucket
    if key.endswith('.dva_export.zip'):
        ofname = "{}/{}/{}.zip".format(settings.MEDIA_ROOT, dv.pk, dv.pk)
        resource.meta.client.download_file(bucket, key, ofname,ExtraArgs={'RequestPayer': 'requester'})
        zipf = zipfile.ZipFile(ofname, 'r')
        zipf.extractall("{}/{}/".format(settings.MEDIA_ROOT, dv.pk))
        zipf.close()
        video_root_dir = "{}/{}/".format(settings.MEDIA_ROOT, dv.pk)
        for k in os.listdir(video_root_dir):
            unzipped_dir = "{}{}".format(video_root_dir, k)
            if os.path.isdir(unzipped_dir):
                for subdir in os.listdir(unzipped_dir):
                    shutil.move("{}/{}".format(unzipped_dir, subdir), "{}".format(video_root_dir))
                shutil.rmtree(unzipped_dir)
                break
        source_zip = "{}/{}.zip".format(video_root_dir, dv.pk)
        os.remove(source_zip)
    else:
        video_root_dir = "{}/{}/".format(settings.MEDIA_ROOT, dv.pk)
        path = "{}/{}/".format(settings.MEDIA_ROOT, dv.pk)
        download_dir(client, resource, key, path, bucket)
        for filename in os.listdir(os.path.join(path, key)):
            shutil.move(os.path.join(path, key, filename), os.path.join(path, filename))
        os.rmdir(os.path.join(path, key))
    with open("{}/{}/table_data.json".format(settings.MEDIA_ROOT, dv.pk)) as input_json:
        video_json = json.load(input_json)
    serializers.import_video_json(dv, video_json, video_root_dir)
    dv.uploaded = True
    dv.save()
    process_next(task_id)
    start.completed = True
    start.seconds = time.time() - start_time
    start.save()


def perform_export(s3_export):
    s3 = boto3.resource('s3')
    if s3_export.region == 'us-east-1':
        s3.create_bucket(Bucket=s3_export.bucket)
    else:
        s3.create_bucket(Bucket=s3_export.bucket, CreateBucketConfiguration={'LocationConstraint': s3_export.region})
    time.sleep(20)  # wait for it to create the bucket
    path = "{}/{}/".format(settings.MEDIA_ROOT, s3_export.video.pk)
    a = serializers.VideoExportSerializer(instance=s3_export.video)
    exists = False
    try:
        s3.Object(s3_export.bucket, '{}/table_data.json'.format(s3_export.key).replace('//', '/')).load()
    except ClientError as e:
        if e.response['Error']['Code'] == "404":
            exists = False
        else:
            raise
    else:
        return -1, "Error key already exists"
    with file("{}/{}/table_data.json".format(settings.MEDIA_ROOT, s3_export.video.pk), 'w') as output:
        json.dump(a.data, output)
    upload = subprocess.Popen(args=["aws", "s3", "sync",'--quiet', ".", "s3://{}/{}/".format(s3_export.bucket, s3_export.key)],cwd=path)
    upload.communicate()
    upload.wait()
    s3_export.completed = True
    s3_export.save()
    return upload.returncode, ""


@app.task(track_started=True, name="backup_video_to_s3")
def backup_video_to_s3(s3_export_id):
    start = TEvent.objects.get(pk=s3_export_id)
    if celery_40_bug_hack(start):
        return 0
    start.started = True
    start.task_id = backup_video_to_s3.request.id
    start.operation = backup_video_to_s3.name
    start.save()
    start_time = time.time()
    returncode, errormsg = perform_export(start)
    if returncode == 0:
        start.completed = True
    else:
        start.errored = True
        start.error_message = errormsg
    start.seconds = time.time() - start_time
    start.save()


@app.task(track_started=True, name="push_video_to_vdn_s3")
def push_video_to_vdn_s3(s3_export_id):
    start = TEvent.objects.get(pk=s3_export_id)
    if celery_40_bug_hack(start):
        return 0
    start.task_id = push_video_to_vdn_s3.request.id
    start.started = True
    start.operation = push_video_to_vdn_s3.name
    start.save()
    start_time = time.time()
    returncode, errormsg = perform_export(start)
    if returncode == 0:
        start.completed = True
    else:
        start.errored = True
        start.error_message = errormsg
    start.seconds = time.time() - start_time
    start.save()


def download_dir(client, resource, dist, local, bucket):
    """
    Taken from http://stackoverflow.com/questions/31918960/boto3-to-download-all-files-from-a-s3-bucket
    :param client:
    :param resource:
    :param dist:
    :param local:
    :param bucket:
    :return:
    """
    paginator = client.get_paginator('list_objects')
    for result in paginator.paginate(Bucket=bucket, Delimiter='/', Prefix=dist, RequestPayer='requester'):
        if result.get('CommonPrefixes') is not None:
            for subdir in result.get('CommonPrefixes'):
                download_dir(client, resource, subdir.get('Prefix'), local, bucket)
        if result.get('Contents') is not None:
            for ffile in result.get('Contents'):
                if not os.path.exists(os.path.dirname(local + os.sep + ffile.get('Key'))):
                    os.makedirs(os.path.dirname(local + os.sep + ffile.get('Key')))
                resource.meta.client.download_file(bucket, ffile.get('Key'), local + os.sep + ffile.get('Key'),
                                                   ExtraArgs={'RequestPayer': 'requester'})


@app.task(track_started=True, name="import_video_from_s3")
def import_video_from_s3(s3_import_id):
    start = TEvent.objects.get(pk=s3_import_id)
    if celery_40_bug_hack(start):
        return 0
    start.started = True
    start.task_id = import_video_from_s3.request.id
    start.operation = import_video_from_s3.name
    start.save()
    start_time = time.time()
    path = "{}/{}/".format(settings.MEDIA_ROOT, start.video.pk)
    logging.info("processing key  {}space".format(start.key))
    if start.key.strip() and (start.key.endswith('.zip') or start.key.endswith('.mp4')):
        fname = 'temp_' + str(time.time()).replace('.', '_') + '_' + str(random.randint(0, 100)) + '.' + \
                start.key.split('.')[-1]
        command = ["aws", "s3", "cp",'--quiet', "s3://{}/{}".format(start.bucket, start.key), fname]
        path = "{}/".format(settings.MEDIA_ROOT)
        download = subprocess.Popen(args=command, cwd=path)
        download.communicate()
        download.wait()
        if download.returncode != 0:
            start.errored = True
            start.error_message = "return code for '{}' was {}".format(" ".join(command), download.returncode)
            start.seconds = time.time() - start_time
            start.save()
            raise ValueError, start.error_message
        handle_downloaded_file("{}/{}".format(settings.MEDIA_ROOT, fname), start.video, fname)
        start.completed = True
        start.seconds = time.time() - start_time
        start.save()
        return
    else:
        create_video_folders(start.video, create_subdirs=False)
        command = ["aws", "s3", "cp",'--quiet', "s3://{}/{}/".format(start.bucket, start.key), '.', '--recursive']
        command_exec = " ".join(command)
        download = subprocess.Popen(args=command, cwd=path)
        download.communicate()
        download.wait()
        if download.returncode != 0:
            start.errored = True
            start.error_message = "return code for '{}' was {}".format(command_exec, download.returncode)
            start.seconds = time.time() - start_time
            start.save()
            raise ValueError, start.error_message
        with open("{}/{}/table_data.json".format(settings.MEDIA_ROOT, start.video.pk)) as input_json:
            video_json = json.load(input_json)
        serializers.import_video_json(start.video, video_json, path)
    start.completed = True
    start.save()
    start.completed = True
    start.seconds = time.time() - start_time
    start.save()


@app.task(track_started=True, name="perform_clustering")
def perform_clustering(cluster_task_id, test=False):
    start = TEvent.objects.get(pk=cluster_task_id)
    if celery_40_bug_hack(start):
        return 0
    start.task_id = perform_clustering.request.id
    start.started = True
    start.operation = perform_clustering.name
    start.save()
    start_time = time.time()
    clusters_dir = "{}/clusters/".format(settings.MEDIA_ROOT)
    if not os.path.isdir(clusters_dir):
        os.mkdir(clusters_dir)
    dc = start.clustering
    fnames = []
    for ipk in dc.included_index_entries_pk:
        k = IndexEntries.objects.get(pk=ipk)
        fnames.append("{}/{}/indexes/{}".format(settings.MEDIA_ROOT, k.video.pk, k.features_file_name))
    cluster_proto_filename = "{}{}.proto".format(clusters_dir, dc.pk)
    c = clustering.Clustering(fnames, dc.components, cluster_proto_filename, m=dc.m, v=dc.v, sub=dc.sub, test_mode=test)
    c.cluster()
    cluster_codes = []
    for e in c.entries:
        cc = ClusterCodes()
        cc.video_id = e['video_primary_key']
        if 'detection_primary_key' in e:
            cc.detection_id = e['detection_primary_key']
            cc.frame_id = Region.objects.get(pk=cc.detection_id).frame_id
        else:
            cc.frame_id = e['frame_primary_key']
        cc.clusters = dc
        cc.coarse = e['coarse']
        cc.fine = e['fine']
        cc.coarse_text = " ".join(map(str, e['coarse']))
        cc.fine_text = " ".join(map(str, e['fine']))
        cc.searcher_index = e['index']
        cluster_codes.append(cc)
    ClusterCodes.objects.bulk_create(cluster_codes)
    c.save()
    dc.completed = True
    dc.save()
    start.completed = True
    start.seconds = time.time() - start_time
    start.save()


@app.task(track_started=True, name="sync_bucket_video_by_id")
def sync_bucket_video_by_id(task_id):
    start = TEvent.objects.get(pk=task_id)
    if celery_40_bug_hack(start):
        return 0
    start.task_id = sync_bucket_video_by_id.request.id
    start.started = True
    start.operation = sync_bucket_video_by_id.name
    start.save()
    start_time = time.time()
    video_id = start.video_id
    args = json.loads(start.arguments_json)
    if settings.MEDIA_BUCKET.strip():
        if 'dirname' in args:
            src = '{}/{}/{}/'.format(settings.MEDIA_ROOT, video_id, args['dirname'])
            dest = 's3://{}/{}/{}/'.format(settings.MEDIA_BUCKET, video_id, args['dirname'])
        else:
            src = '{}/{}/'.format(settings.MEDIA_ROOT, video_id)
            dest = 's3://{}/{}/'.format(settings.MEDIA_BUCKET, video_id)
        command = " ".join(['aws', 's3', 'sync','--quiet', src, dest])
        syncer = subprocess.Popen(['aws', 's3', 'sync','--quiet', '--size-only', src, dest])
        syncer.wait()
        if syncer.returncode != 0:
            start.errored = True
            start.error_message = "Error while executing : {}".format(command)
            start.save()
            return
    else:
        logging.info("Media bucket name not specified, nothing was synced.")
        start.error_message = "Media bucket name is empty".format(settings.MEDIA_BUCKET)
    start.completed = True
    start.seconds = time.time() - start_time
    start.save()
    return


@app.task(track_started=True, name="delete_video_by_id")
def delete_video_by_id(task_id):
    start = TEvent.objects.get(pk=task_id)
    if celery_40_bug_hack(start):
        return 0
    start.task_id = delete_video_by_id.request.id
    start.started = True
    start.operation = delete_video_by_id.name
    start.save()
    start_time = time.time()
    args = json.loads(start.arguments_json)
    video_id = int(args['video_pk'])
    src = '{}/{}/'.format(settings.MEDIA_ROOT, int(video_id))
    args = ['rm','-rf',src]
    command = " ".join(args)
    deleter = subprocess.Popen(args)
    deleter.wait()
    if deleter.returncode != 0:
        start.errored = True
        start.error_message = "Error while executing : {}".format(command)
        start.save()
        return
    if settings.MEDIA_BUCKET.strip():
        dest = 's3://{}/{}/'.format(settings.MEDIA_BUCKET, int(video_id))
        args = ['aws', 's3', 'rm','--quiet','--recursive', dest]
        command = " ".join(args)
        syncer = subprocess.Popen(args)
        syncer.wait()
        if syncer.returncode != 0:
            start.errored = True
            start.error_message = "Error while executing : {}".format(command)
            start.save()
            return
    else:
        logging.info("Media bucket name not specified, nothing was synced.")
        start.error_message = "Media bucket name is empty".format(settings.MEDIA_BUCKET)
    start.completed = True
    start.seconds = time.time() - start_time
    start.save()
    return


@app.task(track_started=True, name="detect_custom_objects")
def detect_custom_objects(task_id):
    """
    :param task_id:
    :return:
    """
    start = TEvent.objects.get(pk=task_id)
    if celery_40_bug_hack(start):
        return 0
    start.task_id = detect_custom_objects.request.id
    start.started = True
    start.operation = detect_custom_objects.name
    start.save()
    start_time = time.time()
    args = json.loads(start.arguments_json)
    video_id = start.video_id
    detector_id = args['detector_pk']
    custom_detector = subprocess.Popen(['fab', 'detect_custom_objects:{},{}'.format(detector_id,video_id)],cwd=os.path.join(os.path.abspath(__file__).split('tasks.py')[0], '../'))
    custom_detector.wait()
    if custom_detector.returncode != 0:
        start.errored = True
        start.error_message = "fab detect_custom_objects failed with return code {}".format(custom_detector.returncode)
        start.seconds = time.time() - start_time
        start.save()
        raise ValueError, start.error_message
    process_next(task_id)
    start.completed = True
    start.seconds = time.time() - start_time
    start.save()
    return 0


@app.task(track_started=True, name="train_yolo_detector")
def train_yolo_detector(task_id):
    """
    :param task_id:
    :return:
    """
    start = TEvent.objects.get(pk=task_id)
    if celery_40_bug_hack(start):
        return 0
    start.task_id = train_yolo_detector.request.id
    start.started = True
    start.operation = train_yolo_detector.name
    start.save()
    start_time = time.time()
    train_detector = subprocess.Popen(['fab', 'train_yolo:{}'.format(start.pk)],cwd=os.path.join(os.path.abspath(__file__).split('tasks.py')[0], '../'))
    train_detector.wait()
    if train_detector.returncode != 0:
        start.errored = True
        start.error_message = "fab train_yolo:{} failed with return code {}".format(start.pk,train_detector.returncode)
        start.seconds = time.time() - start_time
        start.save()
        raise ValueError, start.error_message
    start.completed = True
    start.seconds = time.time() - start_time
    start.save()
    return 0


@app.task(track_started=True,name="update_index")
def update_index(indexer_entry_pk):
    """
    app.send_task('update_index',args=[5,],exchange='broadcast_tasks')
    :param indexer_entry_pk:
    :return:
    """
    print "TESTSTESTSTSTSTSTST"
    logging.info("recieved {}".format(indexer_entry_pk))
    return 0
