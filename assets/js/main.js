
$(document).ready(function () {

	// active nav
	$(".navbar-nav li a");
	const currentLocation = location.href;
	const menuLength = $(".navbar-nav li a").length;
	for (let i = 0; i < menuLength; i++) {
		if ($(".navbar-nav li a")[i].href === currentLocation) {
			$(".navbar-nav li a")[i].className = "active";
		}
	}

	$("[data-trigger]").on("click", function (e) {
		e.preventDefault();
		e.stopPropagation();
		var offcanvas_id = $(this).attr('data-trigger');
		$(offcanvas_id).toggleClass("show");
		$('body').toggleClass("offcanvas-active");
		$(".screen-overlay").toggleClass("show");
	});

	$(".btn-close, .screen-overlay").click(function (e) {
		$(".screen-overlay").removeClass("show");
		$(".mobile-offcanvas").removeClass("show");
		$("body").removeClass("offcanvas-active");
	});

	// Fechando menu depois do click 
	$(document).on('click', '#navbar_main', function (e) {
		if ($(e.target).is('a:not(".dropdown-toggle")')) {
			$(this).collapse('hide');
		}
	});

	if (window.location.hash != "") {
		$('a[href="' + window.location.hash + '"]').click()
	}
});

//footer collapse
$(document).ready(function (e) {
	if ($(window).width() < 768) {
		$('footer .collapse').removeClass('show')
	}
	else {
		$('footer .collapse').addClass('show')
		$("footer .link-collapse").attr("aria-expanded", "true");
	}

})

// captura do json, alimentação do bignumber
$(document).ready(function (e) {
	var mydata = JSON.parse(data);

    // Contagem total
	count = mydata.total_counts

	i = 0;
	while (i < 9) {
		value = Math.floor(count/(10**i))%10;
		$("#total" + i).html(value);
		i++;
	}

    // Contagem de ontem
	count = mydata.yesterday_counts
	i = 0;
	while (i < 9) {
		value = Math.floor(count/(10**i))%10;
		$("#yesterday" + i).html(value);
		i++;
	}

    // Data e horário de atualização
    last_update = mydata.last_update;
    date_time = last_update.split("T");
    dmy = date_time[0].split("-");
    $("#datadate").html(dmy[2] + "/" + dmy[1] + "/" + dmy[0]);
    $("#datatime").html(date_time[1]);
    
})
