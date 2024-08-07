var itemDesign = {};

itemDesign.tweak = function(obj)
{
	//obj.find("img").hide();
	obj.find(".itemContainer").each(function(index) {
		var picWidth = parseInt($(this).find("img").width())
	   $(this).find(".info").css("width", parseInt(obj.width())- picWidth - 70);
	   //console.log("SKIN FIX", $(this).find(".info"), picWidth, parseInt(obj.width())- picWidth - 50);
	});
}

itemDesign.init = function(obj)
{

	itemDesign.tweak(obj);

}