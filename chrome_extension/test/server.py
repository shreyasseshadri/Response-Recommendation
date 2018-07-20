from flask import Flask, request
app = Flask(__name__)
@app.route('/', methods=['POST'])
def result():
    print(eval(request.data.decode('utf-8'))['Query']) # should display 'bar'  
    return "hi bro\n" # response to your request.
if __name__ == '__main__':
    app.run()
