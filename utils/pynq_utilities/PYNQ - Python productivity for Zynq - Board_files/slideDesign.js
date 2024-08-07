var slideDesign = {	
};

slideDesign.buildPic = function(obj){
	obj.find(".itemContainer").each(function(index) {
		$(this).find("img").wrap('<div class="picContainer" />');
		$(this).find("img").load(function() {
			  slideDesign.cropPic($(this).parent());
		});
	});
}

slideDesign.cropPic = function(picContainer){
	
	picContainer.find("img").css("position", "absolute");
	
	
	if ((picContainer.find("img").width()!=0)&&(picContainer.find("img").height()!=0)){
		
		
		var ratioH = parseInt(picContainer.width()) / parseInt(picContainer.find("img").width()) ;
		var ratioV = parseInt(picContainer.height()) / parseInt(picContainer.find("img").height());
		
		
		var ratio = Math.max(ratioH, ratioV);
		
		var newWidth = ratio * parseInt(picContainer.find("img").width())
		var newHeight = ratio * parseInt(picContainer.find("img").height())
		
		var topPos = (parseInt(picContainer.height()) - newHeight) / 2
		var leftPos = (parseInt(picContainer.width()) - newWidth) / 2
		
		
		picContainer.find("img").css("top", topPos)
		picContainer.find("img").css("left", leftPos)
		picContainer.find("img").css("width", newWidth)
		picContainer.find("img").css("height", newHeight)
		
		
		
	}
	
	
	
	
	
}


slideDesign.animateSlide = function(obj, speeed)
{
	/*obj.find(".main").stop().animate({
	    left: 1-parseInt(obj.attr("currentSlide"))*parseInt(obj.width()),
	  }, {
		    duration: speeed,
		    specialEasing: {
		     	 left: 'easeOutQuint'
		    },
		    complete: function() {
		     
		    }
		  });
	*/
	//obj.find(".main").transition({ left: 1-parseInt(obj.attr("currentSlide"))*parseInt(obj.width()), perspective: parseInt(obj.attr("currentSlide"))*parseInt(obj.width()), rotateY: '20deg', duration: 2000, easing: 'snap' });
	
	

	
	
	
	obj.find(".item").each(function(index) {
		
	
	      // this will stop the current animation
	     //$(this).css('-webkit-transition', null);// = null;
		
		//$(this).css3animate({"left": $(this).css('left')}, 0);  
		
		//$(this).css("-webkit-animation-play-state", "paused");
		
		// this will setup the coordinates on the position the animation
	      // was stopped
	     // $(this).css(properties);


		
		
		$(this).css('position', 'absolute')
		var fromCurrent = index - parseInt(obj.attr("currentSlide"));
		var currentSlide = parseInt(obj.attr("currentSlide"));
		//console.log("from current ", fromCurrent, fromCurrent,  fromCurrent * parseInt(obj.width()), parseInt(obj.width()))
		var scale = 1;
		var z3d = 0;
		var rotation = 0;
		if (index > currentSlide){
			$(this).css("z-index", 900-index*50)
			scale = 0.5
			z3d = 2
			rotation = -30;
		}
		if (index < currentSlide){
			$(this).css("z-index", index*50)
			scale = 0.5
			z3d = 2
			rotation = 50
		}
		if (index == currentSlide){
			$(this).css("z-index", 2000)
			scale = 0.7
			z3d = 1;
			rotation = 1
		}

		//console.log()
	    //$(this).style['WebkitTransition'] = null;

		//var $animated = $(this)
		//$(this).css3animate({"left": $(this).css('left'), "rotateY": $(this).css('rotateY')}, 1);
		
		//$(this).stop(true, true).transition({ left: fromCurrent * parseInt(obj.width()/2), perspective: '300px', rotateY: fromCurrent*-30, scale: scale, duration: 750, easing: 'snap' });
		
		console.log("OMG", parseInt(obj.width())/2+"px 200px")
		
		$(this).parent().css('-webkit-perspective-origin', parseInt(obj.width())/2+"px 200px");
		$(this).parent().css('-webkit-perspective', '800px');
		//$(this).parent().css('-webkit-transform', 'translateZ(-1000px)');
		
		$(this).css('-webkit-transform-style', 'preserve-3d');
		$(this).css('-webkit-box-reflect', 'below 1px -webkit-gradient(linear, left top, left bottom, from(transparent), color-stop(0.7, transparent), to(rgba(0,0,0,0.4)))');
		
		$(this).css('-webkit-animation', '2s')
		$(this).css('-webkit-transition', 'all 1.5s cubic-bezier(.25,.91,.12,1)');

		var xPoz = fromCurrent * 200;
	
		
		$(this).find("h2").html(index + " " + currentSlide + " " +fromCurrent)
		
		if (index == currentSlide ) {
			
			$(this).css('-webkit-transform', 'translate3d('+ xPoz +'px, 0px, -300px) rotateY(0deg) ');
			$(this).css('-webkit-filter', 'brightness(0%)');
			//$(this).css('-webkit-transform', 'rotateY(0deg)');
		}
		if (index > currentSlide ) {
			$(this).css('-webkit-transform', 'translate3d('+ parseInt(xPoz+140) +'px, 0px, -800px) rotateY(-50deg) ');
			$(this).css('-webkit-filter', 'brightness(-45%)');
			//$(this).css('-webkit-transform', 'rotateY(30deg)');
		}
		if (index < currentSlide ) {
			$(this).css('-webkit-transform', 'translate3d('+ parseInt(xPoz-140) +'px, 0px, -800px) rotateY(50deg) ');
			$(this).css('-webkit-filter', 'brightness(-45%)');
			//$(this).css('-webkit-transform', 'rotateY(-30deg)');
		}
		
		
		
		
		//$(this).css('-webkit-transform', 'rotateY(60deg) translateZ('+-1000*z3d+'px)');
		//$(this).css('left', fromCurrent * parseInt(obj.width()/2.5))
		//$(this).css('-webkit-filter', 'brightness('+30*z3d+'%)');
		//if (index == currentSlide) {
		//	$(this).css('-webkit-filter', 'brightness(none)');
		//}
	//	$(this).css('scale', scale)
		//transform: rotateY(130deg);
		//-webkit-transform: rotateY(130deg); /* Safari and Chrome */
		//-moz-transform: rotateY(130deg); /* Firefox */
		
	});
	
}

slideDesign.init = function(obj)
{
	
	slideDesign.buildPic(obj)
	
	obj.attr("currentSlide", 0);
	obj.find(".container").append("<div class='slideNav' id='next'><img src='../js/Skins/Item/round_right.png'/></div>")
	obj.find(".container").append("<div class='slideNav' id='prev'><img src='../js/Skins/Item/round_left.png'/></div>")

	obj.find("#next").click(function() {
		obj.attr("currentSlide", parseInt(obj.attr("currentSlide"))+1);
		
		if (parseInt(obj.find(".item").length) <= parseInt(obj.attr("currentSlide"))){
			obj.attr("currentSlide", 0);
		}
		
		slideDesign.animateSlide(obj, 2000)
		
	});
	
	obj.find("#prev").click(function() {
		obj.attr("currentSlide", parseInt(obj.attr("currentSlide"))-1);
		
		if (parseInt(obj.attr("currentSlide")) < 0){
			obj.attr("currentSlide", parseInt(obj.find(".item").length)-1);
		}
		
		slideDesign.animateSlide(obj, 2000)
		
	});
	
	
	obj.find(".container").css("width", "100%");
	obj.find(".container").css("height", "100%");
	
	obj.find(".main").css("width", parseInt(obj.width())*obj.find(".item").length);
	obj.find(".main").css("position", "absolute");
	obj.find(".main").css("top", "0px")
	obj.find(".main").css("left", "0px")
	
	obj.find(".main").css("height", "100%");
	obj.find(".main").css("float", "left");
	obj.find(".main").css("padding", 0);
	obj.find(".main").css("margin", 0);
	
	obj.css("padding", 0);
	obj.css("margin", 0);
	
	
	slideDesign.tweak(obj);

}

slideDesign.tweak = function(obj)
{

	obj.find(".itemContainer").each(function(index) {
		$(this).find(".picContainer").css("width", parseInt(obj.width()));
		$(this).find(".picContainer").css("height", parseInt(obj.height()));
		$(this).find(".picContainer").css("position", "absolute");
		$(this).find(".picContainer").css("left", 0);
		$(this).find(".picContainer").css("top", 0);
		$(this).find(".picContainer").css("z-index", 1);
		$(this).find(".picContainer").css("overflow", "hidden");
		
		slideDesign.cropPic($(this).find(".picContainer"))
		
		//info
		$(this).find(".info").css("position", "absolute");
		$(this).find(".info").css("left", 0);
		$(this).find(".info").css("bottom", 0);
		$(this).find(".info").css("z-index", 2);
		
		$(this).find(".info").css("color", "white");
		$(this).find(".info").css("margin", "20px");
		$(this).find(".info").css("text-shadow", "0px 0px 7px #000");
		
		$(this).find(".info h2").css("color", "white");
		$(this).find(".info h2").css("font-weight", "normal");
		$(this).find(".info h2").css("font-family", obj.attr('vbTitleFontFamily'));
		$(this).find(".info h2").css("font-size", "76px");
		$(this).find(".info h2").css("margin", "0px");
		
		
		
		
		//font-family: OstrichSansRoundedMedium; letter-spacing: 0em; line-height: 100%; font-size: 76px;
		
		
		
		//
	});
	
	obj.find(".item").css("float", "left");
	obj.find(".item").css("width", parseInt(obj.width()))
	obj.find(".item").css("height", parseInt(obj.height()))
	
	obj.find(".itemContainer").css("position", "absolute");
	obj.find(".itemContainer").css("width", "100%");
	obj.find(".itemContainer").css("height", "100%");
	

	obj.find(".slideNav").css("position", "absolute");
	obj.find(".slideNav").css("z-index", 10);
	
	obj.find(".slideNav").css("top", (parseInt(obj.height())-79)/2);

	obj.find("#next").css("right", 0);
	obj.find("#prev").css("left", 0);
	
	
	slideDesign.animateSlide(obj, 0)  

}