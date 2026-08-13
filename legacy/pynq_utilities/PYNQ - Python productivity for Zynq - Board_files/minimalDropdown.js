var minimalDropdown = {};
var widestItem = 0;


minimalDropdown.init = function(obj)
{
	var objContentElement = "";
	//console.log("This is OBJ", obj);
	if (obj){
		if (obj.contentElement){
			this.objContentElement = obj.contentElement
		} else {
			this.objContentElement = obj.parent().parent();
		}
	
	
	//alert("doc ready! 3");
	
	this.objContentElement.find(".minimalDropdown ul.subnav").parent().append("<span></span>"); //Only shows drop down trigger when js is enabled (Adds empty span tag after ul.subnav*)

	this.objContentElement.find(".minimalDropdown ul.topnav li").hover(function() { //When trigger is clicked...

		//find the longest one
		
		if (obj && obj.contentElement){
			$(this).prepend("<div id='bridge' style='position:absolute; left:0px; width:120%; height:160%; background-color: transparent;'></div>")
		} else {
			$(this).prepend("<div id='bridge' style='position:absolute; z-index:-1; left:0px; width:120%; height:160%; background-color: transparent;'></div>")
		}
		
		
		//Following events are applied to the subnav itself (moving subnav up and down)
		$(this).find("ul.subnav").css("width", '400px');
		$(this).find("ul.subnav").css("z-index", '5');
		$(this).find("ul.subnav").show(); //Drop down the subnav on click

		$(this).hover(function() {
			
			//$(this).css('background-color', "red");
		}, function(){
			$(this).find("#bridge").remove();
			$(this).find("ul.subnav").hide(); //When the mouse hovers out of the subnav, move it back up
		});

		
		
		//Following events are applied to the trigger (Hover events for the trigger)
		}).hover(function() {
			
			
			$(this).addClass("subhover"); //On hover over, add class "subhover"
			
			if (obj){
				
				if (obj.contentElement) {
					
					$(this).css('background', obj.vbItemBackgroundColorHover)			
					$(this).css('color', obj.vbItemTextColorHover)
					
				} else {
					
					$(this).css('background', obj.parent().parent().attr('vbItemBackgroundColorHover'))
					$(this).css('color', obj.parent().parent().attr('vbItemTextColorHover'))
					
				}
				
			}

			
			widestItem = 0;
			
			$(this).find("ul.subnav a").each(function(index) {
			    if (widestItem < $(this).width()) {
			    	widestItem = $(this).width()
			    }
			});
			
			//console.log("This is th ewidest item!", widestItem, obj.contentElement.find(".minimalDropdown ul.topnav li").css("width"));
			$(this).find("ul.subnav").css("width", widestItem*1.3 + 25);
			$(this).find("ul.subnav li").css("width", widestItem*1.3 + 25);
			
			
		}, function(){	//On Hover Out
			
			if (obj){
				if (obj.contentElement) {
					
					$(this).css('background', obj.vbItemBackgroundColorNormal)
					$(this).css('color', obj.vbItemTextColorNormal)
					
				} else {
					
					$(this).css('background', obj.parent().parent().attr('vbItemBackgroundColorNormal'))
					$(this).css('color', obj.parent().parent().attr('vbItemTextColorNormal'))
					
				}
			}
			
			
			
			$(this).removeClass("subhover"); //On hover out, remove class "subhover"
			
	});
	
	//design tweaks
	/*
	if (obj.contentElement){
		minimalDropdown.tweak(obj);
		setTimeout(function(){minimalDropdown.tweak(this.objContentElement)}, 1000);
	} else {
		minimalDropdown.tweak2(obj);
		console.log("objContentElement", this.objContentElement);
		setTimeout(function(){minimalDropdown.tweak2(this.objContentElement)}, 1000);
	}
	*/
	
	setTimeout(function(){minimalDropdown.tweak(obj)}, 1000);
	
	if (obj) {
		
		if (obj.contentElement) {
			//do nothing
		} else {
			minimalDropdown.applyCSS(this.objContentElement)
		}
		
	} else {
		//do nothing
	}
	
	
	} else {
		//do nothing
	}
	
	
}

minimalDropdown.tweak = function(obj)
{
	var objElement = "";
	
	if (obj) {
		if (obj.contentElement) {
			this.objElement = obj.contentElement;
		} else {
			
			
		}
	}
	
	if (this.objElement) {
		if (this.objElement.width() > this.objElement.height() ) {
			this.objElement.find(".subnav").css("top", this.objElement.find(".topnav").height());
		} else {
			this.objElement.find(".topnav li").css("width",  parseInt(this.objElement.find(".topnav").width()) - parseInt(this.objElement.find("li").css("padding-left")) - parseInt(this.objElement.find("li").css("padding-right")));
			console.log("deducting padding", parseInt(this.objElement.find(".topnav").width()), parseInt(this.objElement.find("li").css("padding-left")))
			this.objElement.find(".subnav").css("top", 0);
			this.objElement.find(".subnav").css("left", this.objElement.find(".topnav").width());
		}
	
	
	
	this.objElement.find(".minimalDropdown ul.topnav li").css('background', obj.vbItemBackgroundColorNormal)
	this.objElement.find(".minimalDropdown ul.topnav li").css('color', obj.vbItemTextColorNormal)
	this.objElement.find(".minimalDropdown ul.topnav li").css('float', obj.vbAlignment)
	this.objElement.find(".minimalDropdown ul.topnav li a").css('float', obj.vbAlignment)
	this.objElement.find(".minimalDropdown ul.topnav li a").css('direction', obj.vbDirection)
	
	
	
	
	if (this.objElement.width() > this.objElement.height() ) {
	
		
		this.objElement.find(".minimalDropdown ul.topnav").css('width', "1200px")
		
		this.objElement.find(".minimalDropdown ul.topnav li").css('margin-bottom', 0)
		this.objElement.find(".minimalDropdown ul.topnav li").css('margin-right', obj.vbItemSpacing+"px")
		this.objElement.find(".minimalDropdown ul.subnav").css('margin-top', obj.vbItemSpacing+"px")
		this.objElement.find(".minimalDropdown ul.subnav li").css('margin-top', "0px")
		
		this.objElement.find(".minimalDropdown ul.topnav li span").css("margin-top", (this.objElement.find(".topnav").height()-16)/2  - parseInt(this.objElement.find(".topnav").find("li").css("padding-top")))
		this.objElement.find(".minimalDropdown ul.topnav li span").css("background", "url(/js/Skins/Menu/arrow_down.png) no-repeat center top")
		this.objElement.find(".minimalDropdown ul.topnav li span").css("float", "left")
	
	} else {
	
		this.objElement.find(".minimalDropdown ul.topnav li").css('margin-bottom', obj.vbItemSpacing+"px")
		this.objElement.find(".minimalDropdown ul.topnav li").css('margin-right', 0)
		this.objElement.find(".minimalDropdown .subnav").css('margin-left', obj.vbItemSpacing+"px")
			
		this.objElement.find(".minimalDropdown ul.topnav li span").css("margin-top", (this.objElement.find(".minimalDropdown ul.topnav li").height()-16)/2)
		this.objElement.find(".minimalDropdown ul.topnav li span").css("background", "url(/js/Skins/Menu/arrow_left.png) no-repeat center top")
		this.objElement.find(".minimalDropdown ul.topnav li span").css("float", "right")
		
	}
	

	
	}
}


minimalDropdown.applyCSS = function (objElement)
{	

	objElement.find('.website-menu-item a').css('letter-spacing', objElement.attr('vbLetterSpacing'));

	objElement.find('.website-menu-item a').css('font-family', objElement.attr('vbFontFamily'));
	objElement.find('.website-menu-item a').css('font-weight', objElement.attr('vbFontWeight'));
	objElement.find('.website-menu-item a').css('font-style', objElement.attr('vbFontStyle'));
	objElement.find('.website-menu-item a').css('font-size', objElement.attr('vbFontMaxSize') + 'pt');
	
	objElement.find(".minimalDropdown ul.topnav li").css('background', objElement.attr('vbItemBackgroundColorNormal'))
	objElement.find(".minimalDropdown ul.topnav li").css('color', objElement.attr('vbItemTextColorNormal'))
	objElement.find(".minimalDropdown ul.topnav li").css('float', objElement.attr('vbAlignment'))
	objElement.find(".minimalDropdown ul.topnav li a").css('float', objElement.attr('vbAlignment'))
	objElement.find(".minimalDropdown ul.topnav li a").css('direction', objElement.attr('vbDirection'))
	

	objElement.find('.topnav li ul.subnav').css('background', objElement.attr('vbItemBackgroundColorNormal'));

	objElement.find('.website-menu-item').css('padding-top', objElement.attr('vbItemPadding') + 'px');
	objElement.find('.website-menu-item').css('padding-bottom', objElement.attr('vbItemPadding') + 'px');
	//console.log("MENU ITEM PADDING SIZE: ", objElement, objElement.attr('vbItemPadding') + 'px', objElement)
	
	objElement.find('.website-menu-item').css('padding-right', objElement.attr('vbItemPaddingHorizontal') + 'px');
	objElement.find('.website-menu-item').css('padding-left', objElement.attr('vbItemPaddingHorizontal') + 'px');
	
	objElement.find('.website-menu-item').css('text-align', objElement.attr('vbAlignment'));
	
	objElement.find('.textAreaWrapper').css('direction', objElement.attr('vbDirection'));
	objElement.find('.textAreaWrapper').css('float', objElement.attr('vbAlignment'));	
	
	//tweak
	

	
	if (objElement.width() > objElement.height() ) {
		
		objElement.find(".minimalDropdown ul").css('width', '1200px')
		objElement.find(".subnav").css("top", objElement.find(".topnav").height());

		objElement.find(".minimalDropdown ul.topnav li").css('margin-bottom', 0)
		objElement.find(".minimalDropdown ul.topnav li").css('margin-right', objElement.attr('vbItemSpacing')+"px")
		objElement.find(".minimalDropdown ul.subnav").css('margin-top', objElement.attr('vbItemSpacing')+"px")
		
		objElement.find(".minimalDropdown ul.topnav li span").css("top", (objElement.find(".topnav").height()-16)/2  - parseInt(objElement.find(".topnav").find("li").css("padding-top")))
		objElement.find(".minimalDropdown ul.topnav li span").css("background", "url(/js/Skins/Menu/arrow_down.png) no-repeat center top")
		objElement.find(".minimalDropdown ul.topnav li span").css("float", "left")
		
	} else {
		

		objElement.find(".minimalDropdown ul.topnav li").css('margin-bottom', objElement.attr('vbItemSpacing')+"px")
		objElement.find(".minimalDropdown ul.subnav").css('margin-left', objElement.attr('vbItemSpacing')+"px")

		objElement.find(".minimalDropdown ul.topnav li span").css("top", (objElement.find(".minimalDropdown ul.topnav li").height()-16)/2)
		objElement.find(".minimalDropdown ul.topnav li span").css("background", "url(/js/Skins/Menu/arrow_left.png) no-repeat center top")
		objElement.find(".minimalDropdown ul.topnav li span").css("float", "right")
		
		objElement.find("li").css("width",  parseInt(objElement.find(".topnav").width()) - parseInt(objElement.find("li").css("padding-left")) - parseInt(objElement.find("li").css("padding-right")));
		//console.log("deducting padding", parseInt(objElement.find(".topnav").width()), parseInt(objElement.find("li").css("padding-left")))
		objElement.find(".subnav").css("top", 0);
		objElement.find(".subnav").css("left", objElement.find(".topnav").width());
	}
	
	
	
}