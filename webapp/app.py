import os
from flask import Flask, jsonify, request, redirect, url_for, send_from_directory
from sklearn import datasets, svm
from werkzeug.utils import secure_filename

from image_processing.image_parser import get_bounding_boxes, generate_problems

UPLOAD_FOLDER = "tmp"
ALLOWED_EXTENSIONS = set(['png','jpg','jpeg','gif'])

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load Dataset from scikit-learn.
digits = datasets.load_digits()

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/prediction')
def hello():
    clf = svm.SVC(gamma=0.001, C=100.)
    clf.fit(digits.data[:-1], digits.target[:-1])
    prediction = clf.predict(digits.data[-1:])

    return jsonify({'prediction': repr(prediction)})

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'],
                               filename)

@app.route('/',methods=['GET','POST'])
def upload_image():
    if request.method == 'POST':

        # Does post request have a file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)

        file = request.files['file']

        # If user does not select file
        if file.filename  == '':
            flash('No selected file')
            return redirect(request.url)

        # If file exists and is of allowed type
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)

            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))

            bounding_boxes = get_bounding_boxes(filename,"tmp")
            problems = generate_problems(bounding_boxes,filename,"tmp")

            html = "<ul>"

            for problem in problems:
                html += "<li>" + problem.get_question() + " = " + problem.get_response() + "</li>"

            #html += "<li>" + problems[0].get_question() + " = " + problem.get_response() + "</li>"

            html += "</ul>"
            return html

            #return jsonify(bounding_boxes)
            #return redirect(url_for('uploaded_file',filename=filename))


    return '''
    <!doctype html>
    <title>Upload new File</title>
    <h1> Upload new File</h1>
    <form method="post" enctype="multipart/form-data">
        <p><input type="file" name="file">
            <input type="submit" value="upload">

    </form>
    '''


if __name__ == '__main__':
    app.run(host='0.0.0.0')
