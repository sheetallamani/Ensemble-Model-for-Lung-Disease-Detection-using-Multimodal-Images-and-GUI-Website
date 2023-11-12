$(document).ready(function () {
    // Init
    $('.image-section').hide();
    $('.loader').hide();
    $('#result').hide();

    // Upload Preview
    function readURL(input) {
        if (input.files && input.files[0]) {
            var reader = new FileReader();
            reader.onload = function (e) {
                $('#imagePreview').css('background-image', 'url(' + e.target.result + ')');
                $('#imagePreview').hide();
                $('#imagePreview').fadeIn(650);
            }
            reader.readAsDataURL(input.files[0]);
        }
    }
    $("#imageUpload").change(function () {
        $('.image-section').show();
        $('#btn-predict').show();
        $('#btn-predict-2').show();
        $('#result').text('');
        $('#result').hide();
        readURL(this);
    });

    // Predict using Model 1
    $('#btn-predict').click(function () {
        var form_data = new FormData($('#upload-file')[0]);

        // Show loading animation
        $(this).hide();
        $('#btn-predict-2').hide();
        $('.loader').show();

        // Make prediction by calling api /predict
        $.ajax({
            type: 'POST',
            url: '/predict',
            data: form_data,
            contentType: false,
            cache: false,
            processData: false,
            async: true,
            success: function (data) {
                // Get and display the result
                $('.loader').hide();
                $('#btn-predict-2').show();
                $('#result').fadeIn(600);
                $('#result').text('Result is "' + data.result + '" with a percentage of ' + data.percentage + '%');
                console.log('Model 1 Success!');
            },
            
        });
    });

    // Predict using Model 2
    $('#btn-predict-2').click(function () {
        var form_data = new FormData($('#upload-file')[0]);

        // Show loading animation
        $(this).hide();
        $('#btn-predict').hide();
        $('.loader').show();

        // Make prediction by calling api /predict-2
        $.ajax({
            type: 'POST',
            url: '/predict-2',
            data: form_data,
            contentType: false,
            cache: false,
            processData: false,
            async: true,
            success: function (data) {
                // Get and display the result
                $('.loader').hide();
                $('#btn-predict').show();
                $('#result').fadeIn(600);
                $('#result').text(' Model 2 Result:  ' + data);
                console.log('Model 2 Success!');
            },
        });
    });

});
