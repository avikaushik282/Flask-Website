from flask import Flask , render_template
from flask.globals import request
import joblib
import numpy as np
import pickle
app = Flask(__name__)

advert_pre = pickle.load(open('advertpre.pkl','rb'))
mileage_pre = pickle.load(open('milepre.pkl','rb'))
loan_pre = pickle.load(open('loanpre.pkl','rb'))
logist = pickle.load(open('lore.pkl','rb'))
decisiontr = pickle.load(open('dt.pkl','rb'))
seriousvlack = pickle.load(open('sv.pkl','rb'))
rando = pickle.load(open('rf.pkl','rb'))
kano = pickle.load(open('kn.pkl','rb'))
nosfer = pickle.load(open('nb.pkl','rb'))

@app.route('/')
def home():
    return render_template('index.html')
@app.route('/advertise')
def predict3():
    dts = float(request.args.get('dts'))
    aged = float(request.args.get('aged'))
    arainc = float(request.args.get('arainc'))
    diu = float(request.args.get('diu'))
    gendr = float(request.args.get('gendr'))
    inp5 = [[dts,aged,arainc,diu,gendr]]

    h = advert_pre.predict(inp5)
    img3 = '../static/images/happy.gif'
    img4 = '../static/images/sad.gif'
    if h[0] == 1:
        return render_template('index.html', prediction_text6=f"User will click on the advertisement.",result=True,img3=img3)
    else:
        return render_template('index.html', prediction_text6=f"User will not click on the advertisement.",result=True,img3=img4)
@app.route('/mil')
def predict():
    ori = float(request.args.get('ori'))
    cyl = float(request.args.get('cyl'))
    displa = float(request.args.get('displa'))
    horp = float(request.args.get('horp'))
    wt = float(request.args.get('wt'))
    acl = float(request.args.get('acl'))
    yr = float(request.args.get('yr'))
    inp = [[ori,cyl,displa,horp,wt,acl,yr]]

    a = mileage_pre.predict(inp)
    return render_template('index.html', prediction_text1=f"Mileage is {a[0]} km/l")

@app.route('/loan')
def predict1():
    ag = float(request.args.get('ag'))
    dur = float(request.args.get('dur'))
    cam = float(request.args.get('cam'))
    das = float(request.args.get('das'))
    previ = float(request.args.get('previ'))
    vrate = float(request.args.get('vrate'))
    prie = float(request.args.get('prie'))
    con = float(request.args.get('con'))
    eu = float(request.args.get("eu"))
    emp = float(request.args.get('emp'))
    inp1 = [[ag,dur,cam,das,previ,vrate,prie,con,eu,emp]]

    b = loan_pre.predict(inp1)
    img1 = '../static/images/happy.gif'
    img2 = '../static/images/sad.gif'
    if b[0] == 1:
        return render_template('index.html', prediction_text2=f"You can apply for loan.",result=True,img1=img1)
    else:
        return render_template('index.html', prediction_text2=f"You cannot apply for loan.",result=True,img1=img2)

@app.route('/allclassifier')
def predict2():
    pid = float(request.args.get('pid'))
    sx = float(request.args.get('sx'))
    pcla = float(request.args.get('pcla'))
    sis = float(request.args.get('sis'))
    pach = float(request.args.get('pach'))
    fre = float(request.args.get('fre'))
    emba = float(request.args.get('emba'))
    inp2 = [[pid,sx,pcla,sis,pach,fre,emba]]

    f = decisiontr.predict(inp2)
    im1 = '../static/images/survive.gif'
    im2 = '../static/images/notsurvive.gif'
    if f[0] == 1:
        return render_template('index.html', prediction_text3=f"Passanger survived!!",result=True,img4=im1,prediction_text5=f"Random forest is the most accurate model.")
    else:
        return render_template('index.html', prediction_text3=f"Alas! The passanger did not survive",result=True,img4=im2,prediction_text5=f"Random forest is the most accurate model")




if __name__=="__main__":
    app.run(debug=True)