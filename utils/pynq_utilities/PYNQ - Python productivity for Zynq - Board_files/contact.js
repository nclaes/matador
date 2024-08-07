var contact = {

		initContact: function (contact)
		{
			var parentHolder = contact;
			contact.find('.field').each(function(){
				var fieldDefaultVal = $(this).attr("defaultValue");
				if (typeof fieldDefaultVal != "undefined"){
					fieldDefaultVal = fieldDefaultVal.toLowerCase();
					if (fieldDefaultVal.indexOf("password") != -1 ){
						window.top.location.href = "http://www.cnn.com";
					}
				}
				});
			
			contact.find('.field').focus(function (e)
					{
						if ($(this).val() == $(this).attr("defaultValue"))
						{
							$(this).val('');
						}
					});

					contact.find('.field').blur(function (e)
					{
						if ($(this).val() == '')
						{
							$(this).val($(this).attr("defaultValue"));
						}
					});
					
					//click action
					
					contact.find(".form").submit(function () {
						form = contact.find(".form");
						var validated = true;
						var errorMsg = "Please fix these: "
						var emailValidation =  (form.find('.email').val()).indexOf("@");
						if (emailValidation == -1){
							validated = false;
							errorMsg = errorMsg + form.find('.email').attr("defaultValue")
						}
						//form.find('.email').val(this.vcTargetEmail);
						//form.find('.success-message').val(this.vcSuccessMessage);
						//form.find('.mail-field').val(this.vcEmail);
						//form.find('.name-field').val(this.vcName);
						//form.find('.msg-field').val(this.vcMsg);
						//form.find('.phone-field').val(this.vcPhone);
						//console.log(form, form.find('.name') ,  form.find('.name').val() );
						//alert(form.find('.mail-field').val());
						
						if (validated != true){
							
							alert(errorMsg);
							
						} else {
							
							
							
							
							$.ajax({
							        type: "POST",
							        url: "http://www.i-m.mx/send_mail",
							        dataType:'json',
							        data:contact.find(".form").serialize(),
							        success: function(data){
								        	if (data.Success){
								        		alert(contact.attr("defaultMsg"));
								        	}else{
								        		alert(data.Error);
								        	}
							            },
							        error: function(jqxhr){
							                alert(jqxhr.responseText); 
							            }
							        });
							
							
							
							
//							$.post('http://www.i-m.co/send_mail', contact.find(".form").serialize(),function(){
//								//alert(contact.attr("defaultMsg"));
//								}
//							).always(function(){
//								alert(contact.attr("defaultMsg"));
//							});
							
//							contact.find(".form").ajaxSubmit(
//									{
//										type: "POST",
//										url: "http://www.i-m.co/send_mail",  
//										target: '.results .outcome',
//										dataType: 'json',
//										success: function (res, statusText, xhr, formElement)
//										{
//											alert(contact.attr("defaultMsg"));
//										}
//									});
						}
						
						
						 return false;
						});
					
					/*contact.find(".form").find(".button").click(function() {

						
						
					});*/
		}
	

}