#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 26 15:40:45 2018

"""

from flask import Flask ,render_template, flash, request
from flask_wtf import FlaskForm
from wtforms import StringField, SelectField, SubmitField, IntegerField
from flask_wtf.csrf import CSRFProtect
#from wtforms.validators import DataRequired
import pandas as pd
#import json
import os
from gevent import pywsgi

import numpy as np
from sklearn.externals import joblib

csrf = CSRFProtect()

app = Flask(__name__)


class LoginForm(FlaskForm):
    CRS_ELAPSED_TIME = IntegerField('ELAPSED_TIME (min)')
    Airline = SelectField('Airline',choices=[
                                             ('2', 'UNIQUE_CARRIER_AS'),
                                             ('3', 'UNIQUE_CARRIER_B6'),
                                             ('4', 'UNIQUE_CARRIER_DL'),
                                             ('5', 'UNIQUE_CARRIER_EV'),
                                             ('6', 'UNIQUE_CARRIER_F9'),
                                             ('7', 'UNIQUE_CARRIER_HA'),
                                             ('8', 'UNIQUE_CARRIER_NK'),
                                             ('9', 'UNIQUE_CARRIER_OO'),
                                             ('10', 'UNIQUE_CARRIER_UA'),
                                             ('11', 'UNIQUE_CARRIER_VX'),
                                             ('12', 'UNIQUE_CARRIER_WN')])
    OrigAirport = StringField('Origin airport (ex: DTW,ORD...)')
    DestgAirport = StringField('Destination airport (ex: DFW,ATL,PHX...)')
    
    Month = SelectField('Months',\
                          choices=[('1', 'January'), ('2', 'February'),
                                   ('3', 'March')])
    Week = SelectField('Day of week',\
                          choices=[('1', 'Monday'), ('2', 'Tuesday'),
                                   ('3', 'Wednesday'), ('4', 'Thursday'),
                                   ('5', 'Friday'), ('6', 'Saturday'),
                                   ('7', 'Sunday')])
    Distance = SelectField('Distance group',\
                          choices=[('1', 'DISTANCE_GROUP_1'),
                                     ('2', 'DISTANCE_GROUP_2'),
                                     ('3', 'DISTANCE_GROUP_3'),
                                     ('4', 'DISTANCE_GROUP_4'),
                                     ('5', 'DISTANCE_GROUP_5'),
                                     ('6', 'DISTANCE_GROUP_6'),
                                     ('7', 'DISTANCE_GROUP_7'),
                                     ('8', 'DISTANCE_GROUP_8'),
                                     ('9', 'DISTANCE_GROUP_9'),
                                     ('10', 'DISTANCE_GROUP_10'),
                                     ('11', 'DISTANCE_GROUP_11')])
    TimeDep = IntegerField('Time of Departure (hhmm) format')
    TimeArr = IntegerField('Time of Arrival (hhmm) format')
    submit = SubmitField('Compute')


@app.route('/', methods=['GET', 'POST'])
def base_handler():
    form = LoginForm()
    if form.validate_on_submit():
#   Time, Tail, AirlineID, Oairport, destAirport, dep_time, arrival_time, samesate, Month, Week
#    if request.method=='POST':
        result=request.form # recuperation des resultats
        CRS_ELAPSED_TIME = result['CRS_ELAPSED_TIME']
        Airline = result['Airline']
        Oairport = result['OrigAirport']
        destAirport = result['DestgAirport']
        dep_time = int(result['TimeDep'])
        arrival_time = int(result['TimeArr'])
        Month = result['Month']
        Week = result['Week']
        Distance = result['Distance']
        Couple = result['OrigAirport']+'-'+result['DestgAirport']
        
        
        #get variables in the right format for the prediction
        Count_ORIGIN_AIRPORT_ID = list(origin[origin.ORIGIN == Oairport].valeurs)[0]
        Count_DEST_AIRPORT_ID = list(destination[destination.DEST == destAirport].valeurs)[0]
        Count_Couple = list(couples.loc[couples.COUPLE == Couple ].valeurs)[0]
        
        cat = np.array([[int(Airline),int(Month), int(Week),int(Distance)]])
    
        query_cat = pd.DataFrame(cat, columns = ['Airline','Distance', 'Week','Month'])
        
        query_cat = pd.concat([query_cat.drop('Airline', axis=1), pd.get_dummies(query_cat['Airline'])], axis=1)
        query_cat = pd.concat([query_cat.drop('Distance', axis=1), pd.get_dummies(query_cat['Distance'])], axis=1)
        query_cat = pd.concat([query_cat.drop('Week', axis=1), pd.get_dummies(query_cat['Week'], prefix= 'Week')], axis=1)
        query_cat = pd.concat([query_cat.drop('Month', axis=1), pd.get_dummies(query_cat['Month'], prefix= 'Month')], axis=1)
        
        
        num = np.array([[int(CRS_ELAPSED_TIME), int(Count_ORIGIN_AIRPORT_ID), int(Count_DEST_AIRPORT_ID),
                         int(Count_Couple),int(dep_time),int(arrival_time)]])
        
        #Get normalized value for numerical features
        query_num = pd.DataFrame(num, columns = ['Count_ORIGIN_AIRPORT_ID', 'Count_DEST_AIRPORT_ID','Count_Couple', 'dep_time','arrival_time','CRS_ELAPSED_TIME'])
    
       
        #create a data frame with the same columns than X used to learn the model
        query_tot = pd.concat([query_num, query_cat], axis = 1)
        #query_tot=query_tot[~query_tot.columns.duplicated()]
        query = query_tot.reindex(columns=model_columns, fill_value=0)
        
        prediction = SGD.predict(query)

        flash('prediction of your plane delay:{} '.format(prediction))

        
    return render_template('home.html', form=form)



def start_server():   
    app.secret_key = os.urandom(24)
    print("SERVER STARTED")
    port = int(os.environ.get('PORT', 5000))
    pywsgi.WSGIServer(('0.0.0.0', port), app).serve_forever()
    
if __name__ == '__main__':
    SGD = joblib.load('linear_regression.pkl')
    model_columns = joblib.load('model_columns.pkl')
    
    origin = pd.read_csv('origin.csv')
    destination = pd.read_csv('destination.csv')
    couples = pd.read_csv('couples.csv')
    start_server()
