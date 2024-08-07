var slideshow = {
		
		
		rootUrl: "http://d2c8yne9ot06t4.cloudfront.net/static",//http://app.imcreator.com";
		
		currentSlideShowBox: "",
		
		down_x: 0,
		up_x: 0,
		
		paginatorCheck: false,
		
		ScrollToPosition: function (target_to_scroll)
		{
			var kkk = $("."+target_to_scroll);
    		$('html, body').animate({
        		scrollTop: kkk.offset().top
    		}, 2000);
		},
		
		vAlign: function(thiss) {
		            var ah = thiss.height();
		            var ph = thiss.parent().height();
		            if (ah < ph ){
		            	var mh = (ph - ah) / 2;
			            thiss.css('margin-top', mh);
		            } else {
		            	thiss.css('margin-top', 0);
		            }
		            
		    },
		
		fixedPositionArranger: function ()
		{
			
			$(".fixedPos").each(function( index ) {
				
				var attr = $(this).attr('posX');
				if (typeof attr == 'undefined' || attr == false) {
					$(this).css("position", "fixed");
					$(this).attr("posX", $(this).css("left"));
					$(this).attr("posY", $(this).css("top"));
				 
					
					//css += 'top:'+(Math.floor(vb.vbY)+ pageWrapper.offset().top - 60)+'px;'
				}
				
				//css += 'left:'+(Math.floor(vb.vbX)+ $(".page").offset().left)+'px;'
				newXPos = Math.ceil( parseInt($(this).attr("posX")) + Math.ceil(parseInt($(".page").offset().left)) )+"px";
				newYPos = Math.ceil( parseInt($(this).attr("posY")) + Math.ceil(parseInt($(".page").offset().top)) )+"px";
				
				
				if ($(this).hasClass('strechH')){
					newXPos = 0;//Math.ceil( parseInt($(this).attr("posX")) + Math.ceil(parseInt($(".page").offset().left)) )+"px";
				}
				
				$(this).css("left", newXPos)
				$(this).css("top", newYPos)
				
				 //alert($(this).attr("posX"));
				 //$(this).attr("posY") = $(this).css("top");
				 //$(this).css("left", Math.floor(vb.vbX)+ pageWrapper.offset().left )
			});
			
			
			//css += 'left:'+(Math.floor(vb.vbX)+ pageWrapper.offset().left)+'px;'
			//css += 'top:'+(Math.floor(vb.vbY)+ pageWrapper.offset().top - 60)+'px;'
		},
		    
		stretchBackground: function ()
		{
			var bgWidth = parseInt($("#strechedBg").attr("bgWidth"))
			var bgHeight = parseInt($("#strechedBg").attr("bgHeight"))
			
			
			if ($('body').height()-22 < $(".page").height()){
				$("#strechedBg").css("width", parseInt($('body').innerWidth()))
			} else {
				$("#strechedBg").css("width", parseInt($('body').innerWidth()))
			}
			
			if ($('body').innerWidth() < $(".page").width()){
				$("#strechedBg").css("height", parseInt($('body').innerHeight()))
				
			} else {
				$("#strechedBg").css("height", parseInt($('body').innerHeight()))
			}
			
			
			$("#strechedBg").css("overflow", "hidden")
			
			var ratioX =  window.innerWidth / bgWidth
			var ratioY = $(window).height() / bgHeight
			var ratio = Math.max(ratioX, ratioY);
				
			var newWidth = bgWidth * ratio
			var newHeight = bgHeight * ratio

			$("#strechedImg").css("width", newWidth);
			$("#strechedImg").css("height", newHeight);
			
			$("#strechedImg").css("left", window.innerWidth - newWidth);
			$("#strechedImg").css("top", $(window).height() - newHeight);
			
		},
		
		galleryPaging: function (gallery, pagenumber)
		{
	
			var itemsPerPage = Number(gallery.attr("itemsx")) * Number(gallery.attr("itemy"));

			gallery.find(".page-button").attr("src", slideshow.rootUrl + "/images/editor/gallery-page.png");
			gallery.find("#"+pagenumber).attr("src", slideshow.rootUrl + "/images/editor/gallery-page-selected.png");
			
			gallery.children().each(function(xx) {				
				var child = $(this);
				if (child.attr("class") != "paginator"){
					if ( (xx >= (itemsPerPage)*(pagenumber-1)) && (xx < (itemsPerPage)*pagenumber)) {
						child.css("display", "inline")
					} else {
						child.css("display", "none")
					}

					
					
				}
			
			})
		},
		
		stretchLightbox: function ()
		{
			slideshow.initSterch();
			slideshow.stretchBackground();

			
			var windowHeight = $(window).height()
			
			if (window.screen.height){
			
				//console.log("Inside the loop");
				windowHeight  = Math.min((window.screen.height)+140, $(window).height())		
				
				if ((window.orientation == 90) || (window.orientation == -90)){
					windowHeight  = (window.screen.availWidth)
				}
				
			}
			
			
			 var isiPhone = navigator.userAgent.toLowerCase().indexOf("iphone");
			 var isiPad = navigator.userAgent.toLowerCase().indexOf("ipad");
			 var isiPod = navigator.userAgent.toLowerCase().indexOf("ipod");

			
			if((isiPhone > -1) )
			  {
				 // alert("Iphone!");
				  var windowHeight = $(window).height();
			      //Redirect to iPhone Version of the website.
				  
			  } else {
					$(".lightbox").css("height", windowHeight)
					
					$(".lightbox").find(".PictureBox").each(function(index) {
					    var picWidthLightBox = $(this).height();
					    
					    if ((window.orientation == 90) || (window.orientation == -90)){
					    	picWidthLightBox = $(this).height() * 0.85;
						}
					    
					    var picHeightActual = $(this).find('img').height();
					    $(this).find('img').css('top', (picWidthLightBox-picHeightActual)/2)
					  
					});
			  }


		},
		initItemsBox: function (itemsBox)
		{
			$.get("http://localhost:8082/get_folder_html", { id: itemsBox.attr("vmFolderId")},
					   function(data){
					    
					     itemsBox.find(".container").html('<ul class="main">'+data+'</ul>')
					     
					     //init the skin
					     eval(itemsBox.attr("vbItemsPreset")).init(itemsBox);
			});
		},
		
		initGallery: function (gallery)
		{
	
			var itemsPerPage = Number(gallery.attr("itemsx")) * Number(gallery.attr("itemy"));
			
			var numberOfPics = 0;
			
			var zz = 0;
			
			gallery.children().each(function(xx) {				
				var child = $(this);
				child.attr("id", xx);
				if (child.attr("class") != "paginator"){
					
					numberOfPics = numberOfPics + 1;
					
					if ($.browser.msie) {
						child.find(".info-section").hide();
					}
					
					zz = zz+1;

					if (zz == Number(gallery.attr("itemsx"))){
						zz = 0;
						child.css("margin-right", "0px")
					}
					
					if (gallery.attr("itemclick") != "OpenPictureLink"){
						
						child.find("a").attr("target", "_blank");
					}
					
					child.click(function() {
						slideshow.currentSlideShowBox = $(this).closest(".GalleryBox");
						
						if (slideshow.currentSlideShowBox.attr("itemclick") != "None"){
				    		
						
						if (slideshow.currentSlideShowBox.attr("itemclick") != "OpenPictureLink"){
							
							
							
							if (slideshow.currentSlideShowBox.attr("itemClick") == "ShowNextItem"){
								slideshow.StopAutoPlay(slideshow.currentSlideShowBox); //stop autoplay
								slideshow.nextSlide();
								
							} else {
								if (slideshow.currentSlideShowBox.attr("itemclick") != "OpenPictureLink"){
									slideshow.openFullScreen($(this).attr("id"))
								}
							}
							
						}
						}
						
						
						
					}),	
					
					child.hover(function(){
						$(this).find(".info-section").css("opacity", 1);
						if ($.browser.msie) {
							$(this).find(".info-section").show();
						}
					},
					function(){
						$(this).find(".info-section").css("opacity", 0);
						if ($.browser.msie) {
							$(this).find(".info-section").hide();
						}
					})
				}
			})
			
	
			
			if (itemsPerPage < numberOfPics){  // if paging is needed
					
				
				gallery.append("<div class='paginator'><div class='inner'></div></div>");				
				var numberOfPages = Math.ceil(numberOfPics/itemsPerPage)
				
				for (i=1;i<=numberOfPages;i++)
				{
					gallery.find(".paginator").find(".inner").append("<img class='page-button' src='"+slideshow.rootUrl+"/images/editor/gallery-page.png' id='"+i+"'>")
				}
				
				gallery.find(".page-button").click(function() {
					slideshow.galleryPaging(gallery, Number($(this).attr("id")))
					//alert(Number($(this).attr("id")))
				})
			}
			
			
			
			
			
			var itemsPerPage = Number(gallery.attr("itemsx")) * Number(gallery.attr("itemy"));


		},
				
		
		init: function ()
		{
			
			$("#built_using").remove();
			
			//if window is smaller than page - scroll regulary.
			if ( $(".page").height() >  $(window).height()){
					$(".page").css("top", "0px");
					$(".page").css("margin-top", "0px")
			}
			
			//injecting actions
			var i = 0;
			
			//CSS
			
		
			//SLIDESHOW
			
			$(".SlideShowBox").each(function() {

				slideshow.initSlideShow($(this), true);
			})
			
			//PICTURE
			$(".PictureBox").each(function() {
				slideshow.initPicture($(this));
			})
			
			//ITEMSBOX
			$(".ItemsBox").each(function() {
				slideshow.initItemsBox($(this));
			})
			
			//CONTACT
			
			$(".ContactFormBox").each(function() {

				contact.initContact($(this));
			})
			
			
			
			
			//GALLERY
			
			$(".GalleryBox").each(function() {
				var gallery = $(this)
				slideshow.initGallery(gallery)
				slideshow.galleryPaging(gallery, 1);
			})
			
			
			//SKIN MENU BOX
			
			$(".minimalDropdown ").each(function() {
				var skinMenu = $(this)
				minimalDropdown.init(skinMenu)
				//slideshow.galleryPaging(gallery, 1);
			})
			
			//fixed position
			slideshow.fixedPositionArranger();
			
			
			//Orient listener
			window.addEventListener("orientationchange", function() {
				
				//console.log("orient changed")
				
				$('.lightbox').css("-webkit-transform", "translate3d(0,0,0)");
				$('body').css("-webkit-backface-visibility", "hidden");
	
				$('.lightbox').css("-webkit-perspective", 1000);
				
				
				//console.log("refreshing artifacts")
				
				
			}, false);
			
			
			
			//RESIZE
			
			//var strechInterval = setInterval(function(){
			//	slideshow.stretchLightbox();
			//}, 10000);
			slideshow.fixedPositionArranger();
			
			slideshow.initSterch();
			slideshow.stretchBackground();
			
			slideshow.fixedPositionArranger();
			
			$(window).resize(function() {
				
				slideshow.fixedPositionArranger();
				
				slideshow.initSterch();
				
				slideshow.stretchBackground();
				slideshow.stretchLightbox();
				
				
			});
			
			window.onresize = function(event) {
				
				slideshow.fixedPositionArranger();
				
				slideshow.initSterch();
				slideshow.stretchBackground();
				slideshow.stretchLightbox();
				
			}
			
			$(window.parent.document).scroll(function(){				
				
					$(".lightbox").css('position', 'absolute' );
					if( $("html")){
						$(".lightbox").css('top', $("html").scrollTop() );
					} else {
						console.log("no html")
					}
					$(".lightbox").css('left', "0px" );		
			
			});
			
			
			minimalDropdown.init();
			

		},

		
		
		
		initSterch: function() {
			slideshow.vAlign($(".wrapperer"));
			pageLeftPoz = ( $(window).width() - $('.page').width() ) / -2;
			if (pageLeftPoz > 0) { pageLeftPoz = 0};
			var newXpos = pageLeftPoz //parseInt($('.page').css("left"))*-1  //-1*parseInt($(".page").css("margin-left"));
			newXpos = newXpos;
			var windowWidth = Math.floor((Math.max(parseInt($(window).width()), parseInt($(".page").width()))));
			$("body").css("min-width", $(".page").width() )
			if(windowWidth == Math.floor(parseInt($(".page").width())) ){
				$("body").css("overflow-x", "auto")
			} else {
				$("body").css("overflow-x", "hidden")
			}
			$(".strechH").each(function() {
				var strechElement = $(this);
				if (strechElement.hasClass("PictureBox")){
					var actualImgWidth = strechElement.find("img").attr("realwidth")
					var actualImgHeight = strechElement.find("img").attr("realheight")
					var elementHeight = parseInt(strechElement.css("height"))
					var xAspectRatio = windowWidth / actualImgWidth;
					var yAspectRatio = elementHeight / actualImgHeight;
					var actualRatio = 0
					if (xAspectRatio > yAspectRatio){
						actualRatio = xAspectRatio
					} else {
						actualRatio = yAspectRatio
					}
					var cropY = strechElement.find("img").attr("cropY")
					var cropX = strechElement.find("img").attr("cropX")
					if (cropX){
							windowWidth = windowWidth;
							if (strechElement.hasClass("fixedPos")){
							}else {
								strechElement.css("left", newXpos);
							}
							strechElement.css("width", windowWidth);
							var actualImgWidth = strechElement.find("img").attr("realwidth")
							var actualImgHeight = strechElement.find("img").attr("realheight")
							var elementHeight = parseInt($(this).css("height"))
							var xAspectRatio = windowWidth / actualImgWidth;
							var yAspectRatio = elementHeight / actualImgHeight;
							var actualRatio = 0
							if (xAspectRatio > yAspectRatio){
								actualRatio = xAspectRatio
							} else {
								actualRatio = yAspectRatio
							}
							strechElement.find("img").css("width", actualRatio * actualImgWidth )
							strechElement.find("img").css("height", actualRatio * actualImgHeight )
							var yPos = (elementHeight - parseInt(strechElement.find("img").css("height"))) /4
							var xPos = (windowWidth - parseInt(strechElement.find("img").css("width"))) /2
							strechElement.find("img").css("top", yPos )
							strechElement.find("img").css("left", xPos )
							strechElement.find(".inner-visible").css("width", windowWidth )
					} else {
						strechElement.css("width", windowWidth);
						strechElement.css("left", newXpos );
					}
				} else {
					strechElement.css({"width":"100%","left":"0px"});
					$(".wrapperer").append(strechElement)
					//strechElement.css("width", windowWidth);
					//strechElement.css("left", newXpos );
				}
			});
		},
		
		
		initPicture: function(picturePointer) {
			
			if ($.browser.msie) {
				picturePointer.find(".info-section").hide();
			}
			

			picturePointer.hover(function(){
				
				
				
				//like
							if (($(this).attr("social") == "true") && ($(this).width() > 150)) {
								

									var likeTitle = $(this).find(".info-title").text()
									var likeDesc = $(this).find(".description").text()
									
									if (likeTitle == "") {
										likeTitle = $('meta[name=title]').attr("content");	
									}
									
									if (likeDesc == "") {
										likeDesc = $('meta[name=description]').attr("content");	
									}
									
									var likePic = $(this).find("img").attr("src")
									
									if (likeTitle.length > 30){
										likeTitle = likeTitle.substring(0,30)+"..."
									}
									
									
									if (likeDesc.length > 30){
										likeDesc = likeTitle.substring(0,30)+"..."
									}
									
									likePic = likePic.replace("http://", "");
									likePic = encodeURIComponent(likePic);
									likeTitle = encodeURIComponent(likeTitle);
									likeDesc = encodeURIComponent(likeDesc);
									var likeVbid = $("body").attr("vbid");
									var likeDomain = $("body").attr("domain");
									
									
									var likeLink = likeDomain+"/like?var="+likeVbid+"^"+likeDesc+"^"+likeTitle+"^"+likePic;  //"+"&pic="+this.vcImageUrl+"&title="+"yo dudes whats up?";

									$(this).append("<div class='like-container' style='position:absolute; width:200px; height:60px; z-index:9999999999; top:10px; left:10px; overflow:hidden;'><iframe src='//www.facebook.com/plugins/like.php?href="+likeLink+"&amp;send=false&amp;layout=button_count&amp;width=450&amp;show_faces=false&amp;action=like&amp;colorscheme=light&amp;font&amp;height=21&amp;' scrolling='no' frameborder='0' style='border:none; overflow:hidden; width:450px; height:21px;' allowTransparency='true'></iframe></div>");
							
							}
							
							//}
							
									
									
							/**
							 End of dynamic like test
							*/

				//end of like
				
				
				$(this).find(".info-section").css("opacity", 1);	
				if ($.browser.msie) {
					$(this).find(".info-section").show();
				}
			},
			function(){
				$(this).find(".info-section").css("opacity", 0);
				if ($.browser.msie) {
					$(this).find(".info-section").hide();
				}
				
				$(this).find('.like-container').remove();
				
				
				
				//$(this).find(".info-section").css("filter","alpha(opacity=0)");
			})
		},
			
		
			
		
		
		
		initSlideShow: function(slideshowPointer, startAtZero) {
			
			
			slideshowPointer.find(".PictureBox").first().css("z-index", 999);
			
			if (startAtZero == true){
				slideshowPointer.attr("currentSlide", 0);
			}
			
			
			slideshowPointer.append("<div class='paginator'><div class='inner'><img class='background' src='"+slideshow.rootUrl+"/images/editor/slideshow-paginator.png' USEMAP='#slideshow-paginatora' /><span class='text'>1/1</span><a class='prev-button button' href='javascript:void(0);'></a><a class='next-button button' href='javascript:void(0);'></a></div></div>" )
			
			slideshowPointer.find(".next-button").mouseup(function() {
				slideshow.paginatorCheck = true
				slideshow.currentSlideShowBox = $(this).closest(".SlideShowBox");
				slideshow.StopAutoPlay(slideshowPointer); //stop autoplay
				slideshow.nextSlide();
				
			});
			
			slideshowPointer.find(".prev-button").mouseup(function() {
				slideshow.paginatorCheck = true
				slideshow.currentSlideShowBox = $(this).closest(".SlideShowBox");
				slideshow.StopAutoPlay(slideshowPointer); // stop autoplay
				slideshow.prevSlide();
			});
			
			//actions
			
			if(slideshowPointer.attr("paginator")){
				
				
				if (slideshowPointer.attr("paginator") == "Always"){
					slideshowPointer.find(".paginator").css("opacity", 1);
					
				} else {
					slideshowPointer.find(".paginator").css("opacity", 0);
				}
			
			} else {
				
				slideshowPointer.find(".paginator").css("opacity", 0);
				
			}	
			
			
			slideshowPointer.hover(function(){
				
				if($(this).attr("paginator")){
					if ($(this).attr("paginator") == "Hover"){
						$(this).find(".paginator").css("opacity", 1);
						$(this).find(".info-section").css("opacity", 1);
						if ($.browser.msie) {
							$(this).find(".info-section").show();
						}
					}
					
				} else {
					
					$(this).find(".paginator").css("opacity", 1);
					$(this).find(".info-section").css("opacity", 1);
					if ($.browser.msie) {
						$(this).find(".info-section").show();
					}
					
					
					
				}
				

				//like
				if (($(this).attr("social") == "true") && ($(this).width() > 150) ) {
					

						var likeTitle = $(this).find(".info-title").text()
						var likeDesc = $(this).find(".description").text()
						
						if (likeTitle == "") {
							likeTitle = $('meta[name=title]').attr("content");	
						}
						
						if (likeDesc == "") {
							likeDesc = $('meta[name=description]').attr("content");	
						}
						
						var likePic = $(this).find("img").attr("src")
						
						if (likeTitle.length > 30){
							likeTitle = likeTitle.substring(0,30)+"..."
						}
						
						
						if (likeDesc.length > 30){
							likeDesc = likeTitle.substring(0,30)+"..."
						}
						
						likePic = likePic.replace("http://", "");
						likePic = encodeURIComponent(likePic);
						likeTitle = encodeURIComponent(likeTitle);
						likeDesc = encodeURIComponent(likeDesc);
						var likeVbid = $("body").attr("vbid");
						var likeDomain = $("body").attr("domain");
						
						
						var likeLink = likeDomain+"/like?var="+likeVbid+"^"+likeDesc+"^"+likeTitle+"^"+likePic;  //"+"&pic="+this.vcImageUrl+"&title="+"yo dudes whats up?";
						//console.log(likeLink)
						$(this).find(".like-container").remove();
						$(this).find(".paginator").append("<div class='like-container' style='position:absolute; width:200px; height:60px; z-index:9999999999; top:7px; left:13px; overflow:hidden;'><iframe src='//www.facebook.com/plugins/like.php?href="+likeLink+"&amp;send=false&amp;layout=button_count&amp;width=450&amp;show_faces=false&amp;action=like&amp;colorscheme=light&amp;font&amp;height=21&amp;' scrolling='no' frameborder='0' style='border:none; overflow:hidden; width:450px; height:21px;' allowTransparency='true'></iframe></div>");
				
				}
				
				//}
				
						
						
				/**
				 End of dynamic like test
				*/

	//end of like
	
	
				
			},
			function(){
				
				if($(this).attr("paginator")){
					if ($(this).attr("paginator") == "Always"){
						$(this).find(".paginator").css("opacity", 1);
					} else {
						$(this).find(".paginator").css("opacity", 0);	
					}
				} else {
					$(this).find(".paginator").css("opacity", 0);
				}
				
				
				
				
				$(this).find(".info-section").css("opacity", 0);
				if ($.browser.msie) {
					$(this).find(".info-section").hide();
				}
			})
			
			
			
			
			//for each slide
			
			//alert("YAYWTFWTF7");			

			
			slideshowPointer.children().each(function() {
			

				var child = $(this);
				
				
				if ($.browser.msie) {
					child.find(".info-section").hide();
				}
				
				
				if (child.attr("class") != "paginator"){
					
					child.css("position", "absolute");
					child.css("top", "0px");
					child.css("left", "0px");
					
					
					
					
					
					child.click(function() {
						
						
						
					
						
						slideshow.currentSlideShowBox = $(this).closest(".SlideShowBox");
						
						
						if (slideshow.currentSlideShowBox.attr("itemclick") != "None"){
						
							if (slideshow.currentSlideShowBox.attr("itemclick") != "OpenPictureLink"){
								
								if (slideshow.currentSlideShowBox.attr("itemClick") == "ShowNextItem"){
									
									//do nothing
									//slideshow.StopAutoPlay(slideshow.currentSlideShowBox); //stop autoplay
									//slideshow.nextSlide();
									
								} else {
									if (slideshow.currentSlideShowBox.attr("itemclick") != "OpenPictureLink"){
										if (slideshow.attr('class') != 'lightbox'){
											slideshow.openFullScreen($(this).attr("id"))
										}
										
									}
								}
								
							}
						}	
							
						});
								
					
					//TOUCH

					child.mousedown(function(e){
						slideshow.currentSlideShowBox = $(this).closest(".SlideShowBox");
						e.preventDefault();
					    slideshow.down_x = e.pageX;
					  });
						
					child.mouseup(function(e){
						slideshow.up_x = e.pageX;
					    do_work();
					  });
						
					
					child.find('img').unbind('touchstart').bind('touchstart', function(e){
						slideshow.down_x = e.originalEvent.touches[0].pageX;
					  });
						
					child.find('img').unbind('touchmove').bind('touchmove', function(e){
					    e.preventDefault();
					    slideshow.up_x = e.originalEvent.touches[0].pageX;
					  });
						
					child.find('img').unbind('touchend').bind('touchend', function(e){
						slideshow.currentSlideShowBox = $(this).closest(".SlideShowBox");
					    do_work();
					  });

					function do_work()
					{

					  if ((slideshow.down_x - slideshow.up_x) > 50)
					    {
						  slideshow.nextSlide();
					    } else {
					    	if ((slideshow.up_x - slideshow.down_x) > 50)
						    {
						    	 slideshow.prevSlide();
						    } else {
						    	
						    	if ($(".lightbox").children().length != 0){
						    		
						    		$(".lightbox").remove();
						    		$(".blockerr").remove();
						    		
						    		$(".page").css("visibility","visible");
						    		
						    		$("body").css("overflow","auto");
						    			
						    	} else {
						    		//console.log("NO lightbox");
						    		
						    	if (slideshow.currentSlideShowBox.attr("itemclick") != "None"){
						    		
						    		if (slideshow.currentSlideShowBox.attr("itemClick") == "ShowNextItem"){
										
						    			slideshow.StopAutoPlay(slideshowPointer); //stop autoplay
						    			if (slideshow.currentSlideShowBox.attr("itemClick") != "OpenPictureLink"){
						    				slideshow.nextSlide();
						    			}
										
						    		} else {
						    			
						    			if (slideshow.currentSlideShowBox.attr("itemClick") != "OpenPictureLink"){
						    				slideshow.openFullScreen()
						    			}
						    		
						    		}
						    		
						    	}
						    	
						      }

						    }
					    
					    }
					  
					  
					  //slideshow.down_x = 0;
					  //slideshow.up_x = 0;
					    
					  
					}
							
					
					
					
					
					
					
					
					
					
				}
			
			})
			
						
				slideshow.currentSlideShowBox = slideshowPointer;
				slideshow.showSlide();
			
			
			   var autoplayFlag = slideshow.currentSlideShowBox.attr("autoplayf");
			   //alert(autoplayFlag);
			   
			   if (autoplayFlag == "false") {
				  
				//do nothing	   
			   } else {
				   slideshow.StartAutoPlay(slideshowPointer)
			   }
			   
			
			
			
		},
		
		showSlide : function()
		{
			
			var i=0;
			slideshow.currentSlideShowBox.children().each(function() {
			
			paginatorText = (Number(slideshow.currentSlideShowBox.attr("currentSlide"))+1) + "/" + (slideshow.currentSlideShowBox.children().length-1)
			slideshow.currentSlideShowBox.find(".text").text(paginatorText)
				
				var child = $(this);
				if (child.attr("class") != "paginator"){
			
					var currentSlideNum = Number(child.closest(".SlideShowBox").attr("currentSlide"))				
					var nextSlideNum = Number(child.closest(".SlideShowBox").attr("currentSlide"))+1
					var prevSlideNum = Number(child.closest(".SlideShowBox").attr("currentSlide"))-1
				
					
					
					
					if (prevSlideNum == -1) {
						prevSlideNum = slideshow.currentSlideShowBox.children().length-2
					}
					
					if (nextSlideNum == slideshow.currentSlideShowBox.children().length-1){
						nextSlideNum = 0;
					}
					
					
					//console.log("These are the 3: ", prevSlideNum,child.closest(".SlideShowBox").attr("currentSlide"), nextSlideNum)
					
					if ( String(i) ==  child.closest(".SlideShowBox").attr("currentSlide")){
						var originalSrc = child.find("img").attr("originalSrc");
						child.css("opacity", 1)
						
						if (child.find("img").attr("src") == ""){
							child.find("img").attr("src", originalSrc)
						}
						
						child.show();
						child.css("z-index", "999")
					}
					
					
					
					else if ( Number(i) == nextSlideNum ) {
							var originalSrc = child.find("img").attr("originalSrc");
							child.css("opacity", 0)
							if ($.browser.msie) {
								child.hide();
								child.css("z-index", "99")
							}
							//child.find("img").css("opacity", 1)
							if (child.find("img").attr("src") == ""){
								child.find("img").attr("src", originalSrc)
							}
							
							child.show();
							child.css("z-index", "999")
							
							if ($.browser.msie) {
								child.hide();
								child.css("z-index", "99")
							}
					} 
					
					else if ( Number(i) == prevSlideNum) {
						var originalSrc = child.find("img").attr("originalSrc");
						child.css("opacity", 0)
						child.css("z-index", "99")
						if ($.browser.msie) {
								child.hide();
								child.css("z-index", "99")
							}
						//child.find("img").css("opacity", 1)
						if (child.find("img").attr("src") == ""){
							child.find("img").attr("src", originalSrc)
						}						
						
						child.show();	
						if ($.browser.msie) {
							child.hide();
							child.css("z-index", "99")
						}
					
					} else {
						
						//child.find("img").attr("src", "")
						child.hide();
						child.css("z-index", "99")
						
					}
					
					
					
					if ( Number(i) == currentSlideNum ) {
						child.css("z-index", "999")
					} else {

						child.css("z-index", "99")
					}
				
					
					i = i+1;
				}
			
			})
		},
		
		StartAutoPlay: function (slideShowPointer)
		{
			delay = Number(slideShowPointer.attr("delay"))*1000;
			slideShowPointer.autoPlayInterval = setInterval(function () { slideshow.currentSlideShowBox = slideShowPointer; slideshow.nextSlide(); }, delay);
		},

		/**
		Stops autoplay temporarily.
		*/
		StopAutoPlay: function (slideShowPointer)
		{
				clearInterval(slideShowPointer.autoPlayInterval)
		},

		
		
		nextSlide : function ()
		{
			var currSlide = Number(slideshow.currentSlideShowBox.attr("currentSlide"));
			currSlide  = currSlide + 1;
			if (currSlide >= (slideshow.currentSlideShowBox.children().length - 1)){
				currSlide = 0;
			}
			slideshow.currentSlideShowBox.attr("currentSlide", currSlide);
			slideshow.showSlide();
			
		},
		
		prevSlide : function ()
		{
			var currSlide = Number(slideshow.currentSlideShowBox.attr("currentSlide"));
			currSlide  = currSlide - 1;
			if (currSlide < 0){
				currSlide = (slideshow.currentSlideShowBox.children().length - 2);
			}
			slideshow.currentSlideShowBox.attr("currentSlide", currSlide);
			slideshow.showSlide();
			
		},
		
		
		openFullScreen : function (firstSlide)
		{
			
			
			$("body").append("<div class='blockerr' style='z-index:9999999999997; background-color: black; position:fixed; top:-100%; left:-100%; width:200%; height:200%;'></div><div class='lightbox' style='position:fixed; top:0px; left:0px; z-index:9999999999999; background-color:black; width:100%; height:100%;'></div>")
			
			$("body").css("overflow","hidden");
			
			$(".page").css("visibility","hidden");
			
			//document.ontouchmove = function(e){ e.preventDefault(); }
			
			var fullscreenDom = slideshow.currentSlideShowBox.clone( false, false );
		
			fullscreenDom.find(".paginator").remove();
			fullscreenDom.attr("class", "SlideShowBox")
			fullscreenDom.attr("paginator", "Hover")
			
			if (firstSlide == null){
				//fullscreenDom.attr("currentSlide", "0")
			} else{
				fullscreenDom.attr("currentSlide", firstSlide)
			}
			
			
			fullscreenDom.attr("autoplayf", "false");
			
			fullscreenDom.css("width", "100%");
			fullscreenDom.css("height", "100%");
			
			fullscreenDom.css("position", "absolute");
			fullscreenDom.css("top", "0px");
			fullscreenDom.css("left", "0px");
			
			var widthh = $(".lightbox").width();
			var heightt = $(".lightbox").height();
			
			slideshow.resizePic(fullscreenDom, widthh, heightt);
			slideshow.initSlideShow(fullscreenDom, false)
			
			$("body").find(".lightbox").append(fullscreenDom);
			
			$(".lightbox").css('position', 'absolute' );		
			$(".lightbox").css('top', $(window).scrollTop() );
			
			if ( $.browser.msie ) {
				
				$(".lightbox").css('position', 'fixed' );		
				$(".lightbox").css('top', "0px" );
				
				   
			}
			
			$(".lightbox").css('left', '0px' );
			

			
			
			$(".lightbox").mousedown(function(e){
				  //e.preventDefault();
				//slideshow.down_x = e.pageX;
				 // e.stopPropagation();
				  
			 });
			
			
			$(".lightbox").mouseup(function(e){
				
				//console.log("MOUSE UP");
				//console.log(slideshow.down_x);
				//console.log(slideshow.up_x);
				
				 //console.log("*******");
				 
			    if ((slideshow.down_x - slideshow.up_x) > 50)
			    {
				 
			    } else {
			    	if ((slideshow.up_x - slideshow.down_x) > 50)
				    {
				    	
				    } else {
				    	
				    	//console.log(e.)	
				    		
				    	if (slideshow.paginatorCheck == false){
				    	
				    		$(".blockerr").remove();
				    		$(".lightbox").remove();
				    		
				    		$(".page").css("visibility","visible");
				    		
				    		$("body").css("overflow","auto");
				    		
				    	}
				    	
				    	slideshow.paginatorCheck = false;
				    		
				    		
				    }
			    
			    }
			    
			    
			    slideshow.down_x = 0;
			    slideshow.up_x = 0;
			    
			    //alert("MOUSE UP LIGHTBOX")
			    
			    
			});
			

			$(".lightbox").unbind('touchstart').bind('touchstart', function(e){
				  e.preventDefault();
				  
			 });
				
			$(".lightbox").unbind('touchmove').bind('touchmove', function(e){
			    e.preventDefault();
			  
			 });
				
			$(".lightbox").unbind('touchend').bind('touchend', function(e){
				e.preventDefault();			
	
				 //console.log("TOUCH END");
				 //console.log(slideshow.down_x);
				 //console.log(slideshow.up_x);
				 //console.log("*******");
				 
			    if ((slideshow.down_x - slideshow.up_x) > 50)
			    {
				 
			    } else {
			    	if ((slideshow.up_x - slideshow.down_x) > 50)
				    {
				    	
				    } else {
				    	
				    	
				    		$(".blockerr").remove();
				    		$(".lightbox").remove();
				    		
				    		$(".page").css("visibility","visible");
				    		
				    		$("body").css("overflow","auto");
				    		
				    }
			    
			    }
			    
			    
			    slideshow.down_x = 0;
			    slideshow.up_x = 0;
				
			   
			   
			    
			 });
			
			
			slideshow.stretchLightbox();
			

	
		},
		
		
		resizePic : function (fullscreenDom, widthh, heightt )
		{
			
			fullscreenDom.children().each(function() {
				var child = $(this);
				if (child.attr("class") != "paginator"){
					
					child.unbind("click");
					
					var ratio = 1;

					child.css("text-align", "center");
					
					
					ratio1 = heightt*0.9 / Number(child.find("img").height());
					ratio2 = widthh*0.9 / Number(child.find("img").width());

					if (ratio1>ratio2){
						ratio = ratio2
					} else {
						ratio = ratio1
					}
					

					
					//console.log(widthh, ratio, child.find(".PictureBox").width());
					
					childWidth = child.find("img").width();
					childHeight = child.find("img").height();
					
					newChildWidth = (childWidth * ratio);
					newChildHeight = (childHeight * ratio);
					
					var originalPicHeight = Number(child.find("img").attr("realHeight"));
					var originalPicWidth = Number(child.find("img").attr("realWidth"));
					
					
					
					if ((newChildWidth > originalPicWidth) || (newChildHeight > originalPicHeight)){
						newChildWidth = originalPicWidth;
						newChildHeight = originalPicHeight;
					}
					
					//console.log(newChildWidth, newChildHeight)
					
					child.find("img").css("width", newChildWidth+"px");
					child.find("img").css("height", newChildHeight+"px");
					
					
					
					childUrl = child.find("img").attr("src");
					
					var PicFromHighP = childUrl.indexOf("=s");
					
					childUrl = childUrl.split("=s");
					
					var sizeForServe = 0;
					if (newChildWidth > newChildHeight){
						sizeForServe = Math.round(newChildWidth)
					} else {
						sizeForServe = Math.round(newChildHeight)
					}
					
					if (PicFromHighP != -1){
						
					childUrl = childUrl[0]+("=s"+sizeForServe);
					} else {
					childUrl = 	childUrl[0];
					}
					
					child.find("img").attr("src", "" );
					child.find("img").attr("originalSrc", childUrl );
					
					
					child.css("width", widthh);
					child.css("height", heightt);
					
					child.find(".inner-visible").css("width", widthh);
					child.find(".inner-visible").css("height", heightt);
					
					var VertCentering = (heightt - newChildHeight) / 2
					
					child.find("img").css("position", "relative");
					child.find("img").css("top", VertCentering+"px");
					child.find("img").css("left", "0px");
					
				}
			
			})
				
		}

}