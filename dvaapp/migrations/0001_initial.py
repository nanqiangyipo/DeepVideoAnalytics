# -*- coding: utf-8 -*-
# Generated by Django 1.11.3 on 2017-07-21 08:05
from __future__ import unicode_literals

from django.conf import settings
import django.contrib.postgres.fields
from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):

    initial = True

    dependencies = [
        migrations.swappable_dependency(settings.AUTH_USER_MODEL),
    ]

    operations = [
        migrations.CreateModel(
            name='AppliedLabel',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('label_name', models.CharField(max_length=200)),
                ('source', models.CharField(choices=[('UI', 'User Interface'), ('DR', 'Directory Name'), ('AG', 'Algorithm'), ('VD', 'Visual Data Network')], default='UI', max_length=2)),
                ('created', models.DateTimeField(auto_now_add=True, verbose_name='date created')),
            ],
        ),
        migrations.CreateModel(
            name='ClusterCodes',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('fine', django.contrib.postgres.fields.ArrayField(base_field=models.IntegerField(), default=[], size=None)),
                ('coarse', django.contrib.postgres.fields.ArrayField(base_field=models.IntegerField(), default=[], size=None)),
                ('coarse_text', models.TextField(default='')),
                ('fine_text', models.TextField(default='')),
                ('searcher_index', models.IntegerField()),
            ],
        ),
        migrations.CreateModel(
            name='Clusters',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('excluded_index_entries_pk', django.contrib.postgres.fields.ArrayField(base_field=models.IntegerField(), default=[], size=None)),
                ('included_index_entries_pk', django.contrib.postgres.fields.ArrayField(base_field=models.IntegerField(), default=[], size=None)),
                ('train_fraction', models.FloatField(default=0.8)),
                ('algorithm', models.CharField(default='LOPQ', max_length=50)),
                ('indexer_algorithm', models.CharField(max_length=50)),
                ('cluster_count', models.IntegerField(default=0)),
                ('pca_file_name', models.CharField(default='', max_length=200)),
                ('model_file_name', models.CharField(default='', max_length=200)),
                ('components', models.IntegerField(default=64)),
                ('started', models.DateTimeField(auto_now_add=True, verbose_name='date created')),
                ('completed', models.BooleanField(default=False)),
                ('m', models.IntegerField(default=16)),
                ('v', models.IntegerField(default=16)),
                ('sub', models.IntegerField(default=256)),
            ],
        ),
        migrations.CreateModel(
            name='CustomDetector',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('name', models.CharField(max_length=100)),
                ('algorithm', models.CharField(default='', max_length=100)),
                ('model_filename', models.CharField(default='', max_length=200)),
                ('arguments', models.TextField(default='')),
                ('phase_1_log', models.TextField(default='')),
                ('phase_2_log', models.TextField(default='')),
                ('class_distribution', models.TextField(default='')),
                ('class_names', models.TextField(default='')),
                ('frames_count', models.IntegerField(default=0)),
                ('boxes_count', models.IntegerField(default=0)),
                ('trained', models.BooleanField(default=False)),
                ('created', models.DateTimeField(auto_now_add=True, verbose_name='date created')),
            ],
        ),
        migrations.CreateModel(
            name='CustomIndexer',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('name', models.CharField(max_length=100)),
                ('algorithm', models.CharField(default='', max_length=100)),
                ('model_filename', models.CharField(default='', max_length=200)),
                ('input_layer_name', models.CharField(default='', max_length=300)),
                ('embedding_layer_name', models.CharField(default='', max_length=300)),
                ('embedding_layer_size', models.CharField(default='', max_length=300)),
                ('indexer_queue', models.CharField(default='', max_length=300)),
                ('retriever_queue', models.CharField(default='', max_length=300)),
            ],
        ),
        migrations.CreateModel(
            name='DeletedVideo',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('name', models.CharField(default='', max_length=500)),
                ('description', models.TextField(default='')),
                ('url', models.TextField(default='')),
                ('original_pk', models.IntegerField()),
                ('deleter', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, related_name='user_deleter', to=settings.AUTH_USER_MODEL)),
                ('uploader', models.ForeignKey(null=True, on_delete=django.db.models.deletion.CASCADE, related_name='user_uploader', to=settings.AUTH_USER_MODEL)),
            ],
        ),
        migrations.CreateModel(
            name='FederatedQueryResults',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('rank', models.IntegerField()),
                ('server_name', models.CharField(max_length=100)),
                ('algorithm', models.CharField(max_length=100)),
                ('distance', models.FloatField(default=0.0)),
                ('results_metadata', models.TextField(default='')),
                ('results_available', models.BooleanField(default=False)),
                ('result_image_data', models.BinaryField(null=True)),
            ],
        ),
        migrations.CreateModel(
            name='Frame',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('frame_index', models.IntegerField()),
                ('name', models.CharField(max_length=200, null=True)),
                ('subdir', models.TextField(default='')),
                ('h', models.IntegerField(default=0)),
                ('w', models.IntegerField(default=0)),
                ('t', models.FloatField(null=True)),
                ('keyframe', models.BooleanField(default=False)),
                ('segment_index', models.IntegerField(null=True)),
            ],
        ),
        migrations.CreateModel(
            name='IndexEntries',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('features_file_name', models.CharField(max_length=100)),
                ('entries_file_name', models.CharField(max_length=100)),
                ('algorithm', models.CharField(max_length=100)),
                ('detection_name', models.CharField(max_length=100)),
                ('count', models.IntegerField()),
                ('approximate', models.BooleanField(default=False)),
                ('contains_frames', models.BooleanField(default=False)),
                ('contains_detections', models.BooleanField(default=False)),
                ('created', models.DateTimeField(auto_now_add=True, verbose_name='date created')),
                ('indexer', models.ForeignKey(null=True, on_delete=django.db.models.deletion.CASCADE, to='dvaapp.CustomIndexer')),
            ],
        ),
        migrations.CreateModel(
            name='IndexerQuery',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('created', models.DateTimeField(auto_now_add=True, verbose_name='date created')),
                ('count', models.IntegerField(default=20)),
                ('algorithm', models.CharField(default='', max_length=500)),
                ('excluded_index_entries_pk', django.contrib.postgres.fields.ArrayField(base_field=models.IntegerField(), default=[], size=None)),
                ('query_float_vector', django.contrib.postgres.fields.ArrayField(base_field=models.FloatField(), default=[], size=None)),
                ('query_int_vector', django.contrib.postgres.fields.ArrayField(base_field=models.IntegerField(), default=[], size=None)),
                ('results', models.BooleanField(default=False)),
                ('metadata', models.TextField(default='')),
                ('source_filter_json', models.TextField(default='')),
                ('approximate', models.BooleanField(default=False)),
                ('indexer', models.ForeignKey(null=True, on_delete=django.db.models.deletion.CASCADE, to='dvaapp.CustomIndexer')),
            ],
        ),
        migrations.CreateModel(
            name='Query',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('created', models.DateTimeField(auto_now_add=True, verbose_name='date created')),
                ('selected_indexers', django.contrib.postgres.fields.ArrayField(base_field=models.CharField(max_length=30), default=[], size=None)),
                ('results_metadata', models.TextField(default='')),
                ('results_available', models.BooleanField(default=False)),
                ('image_data', models.BinaryField(null=True)),
                ('federated', models.BooleanField(default=False)),
                ('user', models.ForeignKey(null=True, on_delete=django.db.models.deletion.CASCADE, related_name='visua_query_user', to=settings.AUTH_USER_MODEL)),
            ],
        ),
        migrations.CreateModel(
            name='QueryResults',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('rank', models.IntegerField()),
                ('algorithm', models.CharField(max_length=100)),
                ('distance', models.FloatField(default=0.0)),
            ],
        ),
        migrations.CreateModel(
            name='Region',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('region_type', models.CharField(choices=[('A', 'Annotation'), ('D', 'Detection'), ('P', 'Polygon'), ('S', 'Segmentation'), ('T', 'Transform')], max_length=1)),
                ('parent_frame_index', models.IntegerField(default=-1)),
                ('parent_segment_index', models.IntegerField(default=-1, null=True)),
                ('metadata_text', models.TextField(default='')),
                ('metadata_json', models.TextField(default='')),
                ('full_frame', models.BooleanField(default=False)),
                ('x', models.IntegerField(default=0)),
                ('y', models.IntegerField(default=0)),
                ('h', models.IntegerField(default=0)),
                ('w', models.IntegerField(default=0)),
                ('polygon_points_json', models.TextField(default='[]')),
                ('created', models.DateTimeField(auto_now_add=True, verbose_name='date created')),
                ('vdn_key', models.IntegerField(default=-1)),
                ('object_name', models.CharField(max_length=100)),
                ('confidence', models.FloatField(default=0.0)),
                ('materialized', models.BooleanField(default=False)),
                ('png', models.BooleanField(default=False)),
            ],
        ),
        migrations.CreateModel(
            name='Segment',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('segment_index', models.IntegerField()),
                ('start_time', models.FloatField(default=0.0)),
                ('end_time', models.FloatField(default=0.0)),
                ('metadata', models.TextField(default='{}')),
                ('frame_count', models.IntegerField(default=0)),
                ('start_index', models.IntegerField(default=0)),
                ('end_frame', models.ForeignKey(null=True, on_delete=django.db.models.deletion.CASCADE, related_name='segment_end', to='dvaapp.Frame')),
                ('start_frame', models.ForeignKey(null=True, on_delete=django.db.models.deletion.CASCADE, related_name='segment_start', to='dvaapp.Frame')),
            ],
        ),
        migrations.CreateModel(
            name='TEvent',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('event_type', models.CharField(choices=[('V', 'Video'), ('SE', 'S3 export'), ('SI', 'S3 import'), ('CL', 'Clustering'), ('E', 'Export as file')], default='V', max_length=2)),
                ('started', models.BooleanField(default=False)),
                ('completed', models.BooleanField(default=False)),
                ('errored', models.BooleanField(default=False)),
                ('error_message', models.TextField(default='')),
                ('operation', models.CharField(default='', max_length=100)),
                ('created', models.DateTimeField(auto_now_add=True, verbose_name='date created')),
                ('seconds', models.FloatField(default=-1)),
                ('file_name', models.CharField(default='', max_length=200)),
                ('key', models.CharField(default='', max_length=300)),
                ('bucket', models.CharField(default='', max_length=300)),
                ('requester_pays', models.BooleanField(default=False)),
                ('arguments_json', models.TextField(default='{}')),
                ('task_id', models.TextField(null=True)),
                ('clustering', models.ForeignKey(null=True, on_delete=django.db.models.deletion.CASCADE, to='dvaapp.Clusters')),
                ('parent', models.ForeignKey(null=True, on_delete=django.db.models.deletion.CASCADE, to='dvaapp.TEvent')),
            ],
        ),
        migrations.CreateModel(
            name='Tube',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('frame_level', models.BooleanField(default=False)),
                ('start_frame_index', models.IntegerField()),
                ('end_frame_index', models.IntegerField()),
                ('metadata_text', models.TextField(default='')),
                ('metadata_json', models.TextField(default='')),
                ('end_frame', models.ForeignKey(null=True, on_delete=django.db.models.deletion.CASCADE, related_name='end_frame', to='dvaapp.Frame')),
                ('end_region', models.ForeignKey(null=True, on_delete=django.db.models.deletion.CASCADE, related_name='end_region', to='dvaapp.Region')),
                ('source', models.ForeignKey(null=True, on_delete=django.db.models.deletion.CASCADE, to='dvaapp.TEvent')),
                ('start_frame', models.ForeignKey(null=True, on_delete=django.db.models.deletion.CASCADE, related_name='start_frame', to='dvaapp.Frame')),
                ('start_region', models.ForeignKey(null=True, on_delete=django.db.models.deletion.CASCADE, related_name='start_region', to='dvaapp.Region')),
            ],
        ),
        migrations.CreateModel(
            name='VDNDataset',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('response', models.TextField(default='')),
                ('date_imported', models.DateTimeField(auto_now_add=True, verbose_name='date created')),
                ('name', models.CharField(default='', max_length=100)),
                ('created', models.DateTimeField(auto_now_add=True, verbose_name='date created')),
                ('description', models.TextField(default='')),
                ('download_url', models.TextField(default='')),
                ('url', models.TextField(default='')),
                ('aws_requester_pays', models.BooleanField(default=False)),
                ('aws_region', models.TextField(default='')),
                ('aws_bucket', models.TextField(default='')),
                ('aws_key', models.TextField(default='')),
                ('root', models.BooleanField(default=True)),
                ('organization_url', models.TextField()),
                ('parent_local', models.ForeignKey(null=True, on_delete=django.db.models.deletion.CASCADE, to='dvaapp.VDNDataset')),
            ],
        ),
        migrations.CreateModel(
            name='VDNDetector',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('response', models.TextField(default='')),
                ('date_imported', models.DateTimeField(auto_now_add=True, verbose_name='date created')),
                ('name', models.CharField(default='', max_length=100)),
                ('created', models.DateTimeField(auto_now_add=True, verbose_name='date created')),
                ('description', models.TextField(default='')),
                ('download_url', models.TextField(default='')),
                ('url', models.TextField(default='')),
                ('aws_requester_pays', models.BooleanField(default=False)),
                ('aws_region', models.TextField(default='')),
                ('aws_bucket', models.TextField(default='')),
                ('aws_key', models.TextField(default='')),
                ('organization_url', models.TextField()),
            ],
        ),
        migrations.CreateModel(
            name='VDNServer',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('url', models.URLField()),
                ('name', models.CharField(max_length=200)),
                ('last_response_datasets', models.TextField(default='[]')),
                ('last_response_detectors', models.TextField(default='[]')),
                ('last_token', models.CharField(default='', max_length=300)),
            ],
        ),
        migrations.CreateModel(
            name='Video',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('name', models.CharField(default='', max_length=500)),
                ('length_in_seconds', models.IntegerField(default=0)),
                ('height', models.IntegerField(default=0)),
                ('width', models.IntegerField(default=0)),
                ('metadata', models.TextField(default='')),
                ('frames', models.IntegerField(default=0)),
                ('created', models.DateTimeField(auto_now_add=True, verbose_name='date created')),
                ('description', models.TextField(default='')),
                ('uploaded', models.BooleanField(default=False)),
                ('dataset', models.BooleanField(default=False)),
                ('segments', models.IntegerField(default=0)),
                ('url', models.TextField(default='')),
                ('youtube_video', models.BooleanField(default=False)),
                ('query', models.BooleanField(default=False)),
                ('parent_query', models.ForeignKey(null=True, on_delete=django.db.models.deletion.CASCADE, to='dvaapp.Query')),
                ('uploader', models.ForeignKey(null=True, on_delete=django.db.models.deletion.CASCADE, to=settings.AUTH_USER_MODEL)),
                ('vdn_dataset', models.ForeignKey(null=True, on_delete=django.db.models.deletion.CASCADE, to='dvaapp.VDNDataset')),
            ],
        ),
        migrations.AddField(
            model_name='vdndetector',
            name='server',
            field=models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to='dvaapp.VDNServer'),
        ),
        migrations.AddField(
            model_name='vdndataset',
            name='server',
            field=models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to='dvaapp.VDNServer'),
        ),
        migrations.AddField(
            model_name='tube',
            name='video',
            field=models.ForeignKey(null=True, on_delete=django.db.models.deletion.CASCADE, to='dvaapp.Video'),
        ),
        migrations.AddField(
            model_name='tevent',
            name='video',
            field=models.ForeignKey(null=True, on_delete=django.db.models.deletion.CASCADE, to='dvaapp.Video'),
        ),
        migrations.AddField(
            model_name='segment',
            name='video',
            field=models.ForeignKey(null=True, on_delete=django.db.models.deletion.CASCADE, to='dvaapp.Video'),
        ),
        migrations.AddField(
            model_name='region',
            name='event',
            field=models.ForeignKey(null=True, on_delete=django.db.models.deletion.CASCADE, to='dvaapp.TEvent'),
        ),
        migrations.AddField(
            model_name='region',
            name='frame',
            field=models.ForeignKey(null=True, on_delete=django.db.models.deletion.CASCADE, to='dvaapp.Frame'),
        ),
        migrations.AddField(
            model_name='region',
            name='user',
            field=models.ForeignKey(null=True, on_delete=django.db.models.deletion.CASCADE, to=settings.AUTH_USER_MODEL),
        ),
        migrations.AddField(
            model_name='region',
            name='vdn_dataset',
            field=models.ForeignKey(null=True, on_delete=django.db.models.deletion.CASCADE, to='dvaapp.VDNDataset'),
        ),
        migrations.AddField(
            model_name='region',
            name='video',
            field=models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to='dvaapp.Video'),
        ),
        migrations.AddField(
            model_name='queryresults',
            name='detection',
            field=models.ForeignKey(null=True, on_delete=django.db.models.deletion.CASCADE, to='dvaapp.Region'),
        ),
        migrations.AddField(
            model_name='queryresults',
            name='frame',
            field=models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to='dvaapp.Frame'),
        ),
        migrations.AddField(
            model_name='queryresults',
            name='indexerquery',
            field=models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to='dvaapp.IndexerQuery'),
        ),
        migrations.AddField(
            model_name='queryresults',
            name='query',
            field=models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to='dvaapp.Query'),
        ),
        migrations.AddField(
            model_name='queryresults',
            name='video',
            field=models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to='dvaapp.Video'),
        ),
        migrations.AddField(
            model_name='indexerquery',
            name='parent_query',
            field=models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to='dvaapp.Query'),
        ),
        migrations.AddField(
            model_name='indexerquery',
            name='user',
            field=models.ForeignKey(null=True, on_delete=django.db.models.deletion.CASCADE, to=settings.AUTH_USER_MODEL),
        ),
        migrations.AddField(
            model_name='indexentries',
            name='source',
            field=models.ForeignKey(null=True, on_delete=django.db.models.deletion.CASCADE, to='dvaapp.TEvent'),
        ),
        migrations.AddField(
            model_name='indexentries',
            name='video',
            field=models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to='dvaapp.Video'),
        ),
        migrations.AddField(
            model_name='frame',
            name='video',
            field=models.ForeignKey(null=True, on_delete=django.db.models.deletion.CASCADE, to='dvaapp.Video'),
        ),
        migrations.AddField(
            model_name='federatedqueryresults',
            name='query',
            field=models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to='dvaapp.Query'),
        ),
        migrations.AddField(
            model_name='federatedqueryresults',
            name='user',
            field=models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to=settings.AUTH_USER_MODEL),
        ),
        migrations.AddField(
            model_name='customdetector',
            name='source',
            field=models.ForeignKey(null=True, on_delete=django.db.models.deletion.CASCADE, to='dvaapp.TEvent'),
        ),
        migrations.AddField(
            model_name='customdetector',
            name='vdn_detector',
            field=models.ForeignKey(null=True, on_delete=django.db.models.deletion.CASCADE, to='dvaapp.VDNDetector'),
        ),
        migrations.AddField(
            model_name='clustercodes',
            name='clusters',
            field=models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to='dvaapp.Clusters'),
        ),
        migrations.AddField(
            model_name='clustercodes',
            name='detection',
            field=models.ForeignKey(null=True, on_delete=django.db.models.deletion.CASCADE, to='dvaapp.Region'),
        ),
        migrations.AddField(
            model_name='clustercodes',
            name='frame',
            field=models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to='dvaapp.Frame'),
        ),
        migrations.AddField(
            model_name='clustercodes',
            name='video',
            field=models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to='dvaapp.Video'),
        ),
        migrations.AddField(
            model_name='appliedlabel',
            name='frame',
            field=models.ForeignKey(null=True, on_delete=django.db.models.deletion.CASCADE, to='dvaapp.Frame'),
        ),
        migrations.AddField(
            model_name='appliedlabel',
            name='region',
            field=models.ForeignKey(null=True, on_delete=django.db.models.deletion.CASCADE, to='dvaapp.Region'),
        ),
        migrations.AddField(
            model_name='appliedlabel',
            name='segment',
            field=models.ForeignKey(null=True, on_delete=django.db.models.deletion.CASCADE, to='dvaapp.Segment'),
        ),
        migrations.AddField(
            model_name='appliedlabel',
            name='tube',
            field=models.ForeignKey(null=True, on_delete=django.db.models.deletion.CASCADE, to='dvaapp.Tube'),
        ),
        migrations.AddField(
            model_name='appliedlabel',
            name='video',
            field=models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to='dvaapp.Video'),
        ),
        migrations.AlterUniqueTogether(
            name='segment',
            unique_together=set([('video', 'segment_index')]),
        ),
        migrations.AlterUniqueTogether(
            name='indexentries',
            unique_together=set([('video', 'features_file_name')]),
        ),
        migrations.AlterUniqueTogether(
            name='frame',
            unique_together=set([('video', 'frame_index')]),
        ),
        migrations.AlterUniqueTogether(
            name='clustercodes',
            unique_together=set([('searcher_index', 'clusters')]),
        ),
        migrations.AlterIndexTogether(
            name='clustercodes',
            index_together=set([('clusters', 'searcher_index')]),
        ),
    ]
