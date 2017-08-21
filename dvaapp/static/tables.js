function InitializeTables(){

    // // var language;

    // var language = window.navigator.language;
    // //IE6
    // var language = <!--[if IE6]>window.navigator.userLanguage<![endif]-->;

  
    $('.dataTables').dataTable({
        responsive: true,
        // "bPaginate": false,
        'oLanguage':'Chinese'

    });
    $('.dataTables-dict').dataTable({
        responsive: true,
        "bPaginate": false,
        'oLanguage':'Chinese'
    });
    $('.dataTables-nofilter').dataTable({
        responsive: true,
        "bPaginate": false,
        "bFilter": false,
        'oLanguage':'Chinese'
    });
}
