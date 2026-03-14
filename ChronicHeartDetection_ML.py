import os
import threading
import numpy as np
import tkinter as tk
from tkinter import filedialog, messagebox

import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.signal import spectrogram

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix

from python_speech_features import mfcc
import wfdb

from datetime import datetime
from reportlab.platypus import SimpleDocTemplate, Paragraph, Image, Spacer
from reportlab.lib.styles import getSampleStyleSheet


# ---------------- FEATURE EXTRACTION ----------------

def extract_features(audio, rate):

    mfcc_feat = mfcc(audio, rate, numcep=13)

    return np.hstack((np.mean(mfcc_feat, axis=0),
                      np.std(mfcc_feat, axis=0)))


# ---------------- TRAIN MODEL ----------------

def train_model():

    folder = filedialog.askdirectory(title="Select Dataset Folder")

    if not folder:
        return

    status_label.config(text="Training model...")

    def job():

        global model, accuracy, sensitivity, specificity, cm

        X=[]
        y=[]
        files=[]

        for root_dir, dirs, file_list in os.walk(folder):

            for file in file_list:

                if file.endswith(".wav") or file.endswith(".dat"):

                    files.append(os.path.join(root_dir,file))


        total=len(files)

        for i,path in enumerate(files):

            try:

                if path.endswith(".wav"):
                    rate,audio=wavfile.read(path)

                else:
                    record=path.replace(".dat","")
                    signals,fields=wfdb.rdsamp(record)
                    audio=signals.flatten()
                    rate=fields["fs"]

                if len(audio.shape)>1:
                    audio=audio.mean(axis=1)

                audio=audio/np.max(np.abs(audio))

                feat=extract_features(audio,rate)

                X.append(feat)

                if i < total/2:
                    y.append(0)
                else:
                    y.append(1)

            except:
                continue


        X=np.array(X)
        y=np.array(y)

        X_train,X_test,y_train,y_test=train_test_split(
            X,y,test_size=0.2,random_state=42
        )

        model=RandomForestClassifier(n_estimators=100)

        model.fit(X_train,y_train)

        y_pred=model.predict(X_test)

        accuracy=accuracy_score(y_test,y_pred)

        cm=confusion_matrix(y_test,y_pred)

        tn,fp,fn,tp=cm.ravel()

        sensitivity=tp/(tp+fn)
        specificity=tn/(tn+fp)

        status_label.config(text="Model trained successfully")

    threading.Thread(target=job).start()


# ---------------- ANALYZE SOUND ----------------

def analyze_sound():

    global report_data, result, confidence

    if model is None:
        messagebox.showwarning("Train model first")
        return

    file=filedialog.askopenfilename(
        filetypes=[("Audio","*.wav *.dat")]
    )

    if not file:
        return


    result_card.pack(pady=25)


    if file.endswith(".wav"):
        rate,audio=wavfile.read(file)

    else:
        record=file.replace(".dat","")
        signals,fields=wfdb.rdsamp(record)
        audio=signals.flatten()
        rate=fields["fs"]


    if len(audio.shape)>1:
        audio=audio.mean(axis=1)

    audio=audio/np.max(np.abs(audio))

    feat=extract_features(audio,rate).reshape(1,-1)

    prediction=model.predict(feat)

    probs=model.predict_proba(feat)

    normal_prob=probs[0][0]*100
    abnormal_prob=probs[0][1]*100

    confidence=max(normal_prob,abnormal_prob)

    result="NORMAL" if prediction[0]==0 else "ABNORMAL"

    color="green" if result=="NORMAL" else "red"

    diagnosis_label.config(text=f"Diagnosis : {result}",fg=color)
    confidence_label.config(text=f"Confidence : {confidence:.2f}%")
    accuracy_label.config(text=f"Accuracy : {accuracy:.2f}")
    sensitivity_label.config(text=f"Sensitivity : {sensitivity:.2f}")
    specificity_label.config(text=f"Specificity : {specificity:.2f}")


    # -------- GRAPH DASHBOARD --------

    fig,axs=plt.subplots(2,2,figsize=(12,8))

    axs[0,0].plot(audio)
    axs[0,0].set_title("Waveform")

    f,t,Sxx=spectrogram(audio,rate)

    axs[0,1].pcolormesh(t,f,Sxx)
    axs[0,1].set_title("Spectrogram")

    axs[1,0].bar(["Normal","Abnormal"],[normal_prob,abnormal_prob])
    axs[1,0].set_title("Prediction")

    axs[1,1].imshow(cm,cmap="Blues")
    axs[1,1].set_title("Confusion Matrix")

    for i in range(2):
        for j in range(2):
            axs[1,1].text(j,i,cm[i,j],ha="center",va="center")

    plt.tight_layout()

    plt.savefig("dashboard.png")

    plt.show()


    report_data=f"""
Diagnosis : {result}

Confidence : {confidence:.2f} %

Accuracy : {accuracy:.2f}

Sensitivity : {sensitivity:.2f}

Specificity : {specificity:.2f}
"""


# ---------------- DOWNLOAD REPORT ----------------

def download_report():

    global report_data

    if report_data == "":
        messagebox.showwarning("Warning","Analyze a heart sound first")
        return


    desktop=os.path.join(os.path.expanduser("~"),"Desktop")

    filename=f"Heart_Report_{datetime.now().strftime('%H%M%S')}.pdf"

    path=os.path.join(desktop,filename)


    styles=getSampleStyleSheet()

    story=[]

    story.append(
        Paragraph("Heart Sound Diagnostic Report",styles["Title"])
    )

    story.append(Spacer(1,20))

    story.append(
        Paragraph(report_data.replace("\n","<br/>"),styles["Normal"])
    )

    story.append(Spacer(1,20))

    if os.path.exists("dashboard.png"):
        story.append(Image("dashboard.png",width=500,height=320))


    pdf=SimpleDocTemplate(path)

    pdf.build(story)

    messagebox.showinfo("Report Saved",f"Report saved on Desktop:\n{filename}")


# ---------------- UI ----------------

root=tk.Tk()

root.title("Heart Sound Analysis Dashboard")

root.geometry("750x520")

root.configure(bg="#F4F6F7")


header=tk.Frame(root,bg="#0B5345",height=70)
header.pack(fill="x")

title=tk.Label(header,text="Heart Sound Analysis System",
font=("Segoe UI",22,"bold"),bg="#0B5345",fg="white")
title.pack(pady=15)


button_frame=tk.Frame(root,bg="#F4F6F7")
button_frame.pack(pady=25)


train_btn=tk.Button(button_frame,text="Train Model",width=18,height=2,
bg="#17A589",fg="white",command=train_model)
train_btn.grid(row=0,column=0,padx=15)


analyze_btn=tk.Button(button_frame,text="Analyze Heart Sound",width=18,height=2,
bg="#2E86C1",fg="white",command=analyze_sound)
analyze_btn.grid(row=0,column=1,padx=15)


report_btn=tk.Button(button_frame,text="Download Report",width=18,height=2,
bg="#8E44AD",fg="white",command=download_report)
report_btn.grid(row=0,column=2,padx=15)


status_label=tk.Label(root,text="Status : Waiting for training",bg="#F4F6F7")
status_label.pack()


# -------- RESULT PANEL --------

result_card=tk.Frame(root,bg="white",bd=2,relief="ridge",padx=30,pady=20)


title2=tk.Label(result_card,text="Analysis Result",
font=("Segoe UI",18,"bold"),fg="#1F618D",bg="white")
title2.pack(pady=10)


diagnosis_label=tk.Label(result_card,text="Diagnosis :",font=("Segoe UI",12,"bold"),bg="white")
diagnosis_label.pack(anchor="w")

confidence_label=tk.Label(result_card,text="Confidence :",font=("Segoe UI",12),bg="white")
confidence_label.pack(anchor="w")

accuracy_label=tk.Label(result_card,text="Accuracy :",font=("Segoe UI",12),bg="white")
accuracy_label.pack(anchor="w")

sensitivity_label=tk.Label(result_card,text="Sensitivity :",font=("Segoe UI",12),bg="white")
sensitivity_label.pack(anchor="w")

specificity_label=tk.Label(result_card,text="Specificity :",font=("Segoe UI",12),bg="white")
specificity_label.pack(anchor="w")


model=None
report_data=""

root.mainloop()