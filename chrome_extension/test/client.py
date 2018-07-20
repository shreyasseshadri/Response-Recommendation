import requests
r = requests.post("http://127.0.0.1:5000/", data={'foo': 'bar'})
# And done.
print(r.text) # displays the result bodY
