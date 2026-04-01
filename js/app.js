$(document).ready(function () {
    // Storing checked boxes in localStorate
    let key = window.location.href + "-checked";
    let checkbox_checked = JSON.parse(localStorage.getItem(key));
    if (checkbox_checked == null) {
        checkbox_checked = {};
    }

    let step_id = 0;
    $(".step").each(function () {
        // Assign an id to each step
        $(this).attr("id", "step-" + step_id);
        checked = '';
        if (checkbox_checked[step_id]) {
            checked = "checked='checked'";
            $(this).addClass("done");
        }

        // Append a checkbox before
        $(this).before("<input type='checkbox' class='step-checkbox' id='step-checkbox-" + step_id + "' " + checked + " />");

        step_id++;
    });

    $(".step-checkbox").change(function () {
        let step_id = $(this).attr("id").split("-")[2];
        let step = $("#step-" + step_id);
        checkbox_checked[step_id] = $(this).is(":checked");
        localStorage.setItem(key, JSON.stringify(checkbox_checked));

        if ($(this).is(":checked")) {
            step.addClass("done");
        } else {
            step.removeClass("done");
        }
    });

    let clean_counter = 0;
    $(document).keydown(function (e) {
        if (e.keyCode == 161) {
            clean_counter++;
            if (clean_counter == 3) {
                checkbox_checked = {};
                localStorage.setItem(key, JSON.stringify(checkbox_checked));
                $(".step").removeClass("done");
                $(".step-checkbox").prop("checked", false);
            }
        } else {
            clean_counter = 0;
        }
    });
});