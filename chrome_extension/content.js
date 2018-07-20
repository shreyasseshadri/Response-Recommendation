chrome.runtime.onMessage.addListener(
  function(request, sender, sendResponse) {
    if( request.message === "clicked_browser_action" ) {
      
       var req = new XMLHttpRequest();
        req.responseType='text';
        req.open("POST", "http://127.0.0.1:5000/", true);
        req.setRequestHeader("Content-Type", "application/json;charset=UTF-8");
        // request.send(JSON.stringify({denny: 1, mark: 1, johnny: 0}));
        req.send(JSON.stringify({Query:window.getSelection().toString()}));

        req.onload=function(){
            if(req.readyState === 4)
              {
                console.log(req.responseText);
              }
        }
    }

  }
);
