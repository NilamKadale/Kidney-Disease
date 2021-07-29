# import the necessary packages
from flask import Flask, render_template, redirect, url_for, request,session,Response
from werkzeug import secure_filename
from supportFile import predict
import os
import cv2
import pickle
from sklearn.ensemble import RandomForestClassifier

app = Flask(__name__)

app.secret_key = '1234'
app.config["CACHE_TYPE"] = "null"
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0

@app.route('/', methods=['GET', 'POST'])
def landing():
	return render_template('home.html')

@app.route('/home', methods=['GET', 'POST'])
def home():
	return render_template('home.html')

@app.route('/info', methods=['GET', 'POST'])
def info():
	return render_template('info.html')

@app.route('/input', methods=['GET', 'POST'])
def input():	
	classifier = pickle.load(open('kidney.pkl', 'rb'))
	if request.method == 'POST':
		age = 	float(request.form['age'])
		bp = 	float(request.form['bp'])
		sg = 	float(request.form['sg'])
		al = 	float(request.form['al'])
		su = 	float(request.form['su'])
		rbc = 	0 if request.form['rbc'] == 'normal' else 1  
		pc = 	0 if request.form['pc'] == 'normal' else 1 
		pcc = 	0 if request.form['pcc']  == 'notpresent' else 1 
		ba = 	0 if request.form['ba'] == 'notpresent' else 1 
		bgr = 	float(request.form['bgr'])
		bu = 	float(request.form['bu'])
		sc = 	float(request.form['sc'])
		sod = 	float(request.form['sod'])
		pot = 	float(request.form['pot'])
		hemo = 	float(request.form['hemo'])
		pcv = 	float(request.form['pcv'])
		wc = 	float(request.form['wc'])
		rc = 	float(request.form['rc'])
		htn = 	0 if request.form['htn'] == 'no' else 1  
		dm = 	0 if request.form['dm'] == 'no' else 1 
		cad = 	0 if request.form['cad'] == 'no' else 1 
		appet = 0 if request.form['appet'] == 'poor' else 1
		pe = 	0 if request.form['pe'] == 'no' else 1 
		ane = 	0 if request.form['ane'] == 'no' else 1

		x_test = [age,bp,al,su,rbc,pc,pcc,ba,bgr,bu,sc,pot,wc,htn,dm,cad,pe,ane]
		pred = classifier.predict([x_test])

		result = "Chronic Kidney Disease Detected" if pred[0] == 0 else "Normal Kidney"
		return render_template('input.html',age=age,bp=bp,sg=sg,al=al,su=su,rbc=rbc,pc=pc,pcc=pcc,ba=ba,
		bgr=bgr,bu=bu,sc=sc,sod=sod,pot=pot,hemo=hemo,pcv=pcv,wc=wc,rc=rc,htn=htn,dm=dm,
		cad=cad,appet=appet,pe=pe,ane=ane,result=result)

	return render_template('input.html')



@app.route('/image', methods=['GET', 'POST'])
def image():
	if request.method == 'POST':
		if request.form['sub']=='Upload':
			savepath = r'upload/'
			photo = request.files['photo']
			photo.save(os.path.join(savepath,(secure_filename(photo.filename))))
			image = cv2.imread(os.path.join(savepath,secure_filename(photo.filename)))
			cv2.imwrite(os.path.join("static/images/","test_image.jpg"),image)
			return render_template('image.html')
		elif request.form['sub'] == 'Test':
			result = predict()
			return render_template('image.html',result=result)
	return render_template('image.html')

# No caching at all for API endpoints.
@app.after_request
def add_header(response):
	# response.cache_control.no_store = True
	response.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, post-check=0, pre-check=0, max-age=0'
	response.headers['Pragma'] = 'no-cache'
	response.headers['Expires'] = '-1'
	return response


if __name__ == '__main__':
	app.run(host='0.0.0.0', debug=True, threaded=True)
