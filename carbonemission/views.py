from django.shortcuts import render
from django.http import HttpResponse
import pandas as pd
import numpy as np
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.offline import plot
from plotly.graph_objs import Scatter
import os
import json
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from plotly.subplots import make_subplots
from fbprophet import Prophet
import pickle

def home(request):
    world = pd.read_csv('carbonemission/meatcattleworldco.csv')
    data = dict(
        type = 'choropleth',
        colorscale='Reds',
        # reversescale = True,
        locations = world['Area'],
        locationmode = "country names",
        z = world['Value'],
        text = world['Area'],
        colorbar = {'title' : 'CO2 Emission in Meat Industry'},
        # wscale=False
        # autocolorscale=True
      ) 

    layout = dict(title = 'CO2 Emission in Meat Industry',margin={"r":35,"t":35,"l":35,"b":35},
                geo = dict(showframe = False,projection = {'type':'natural earth'})
             )
    choromap = go.Figure(data = [data],layout = layout)
    
    
    #fig5 = iplot(choromap)
    fig5 = choromap.to_html()
    df2000 = pd.read_csv('carbonemission/dataset2000.csv')
    df2017 = pd.read_csv('carbonemission/dataset2017.csv')
    fig6 = make_subplots(1, 2, specs=[[{'type':'domain'}, {'type':'domain'}]],
                        subplot_titles=['2000', '2017'])
    fig6.add_trace(go.Pie(labels=df2000['Item'], values=df2000['Value'], scalegroup='one',
                        name="CO2 2000"), 1, 1)
    fig6.add_trace(go.Pie(labels=df2017['Item'], values=df2017['Value'], scalegroup='one',
                        name="CO2 2000"), 1, 2)
    fig6.update_layout(title_text='Carbon Emission in Meat Industry')
    plot_div6 = plot(fig6, output_type='div', include_plotlyjs=False)
    context = {'fig5': fig5,'plot_div6': plot_div6}    
    return render(request,'index.html', context)

def input_visualize(request):    
    return render(request,'input_visualize.html')
def A_InputVisualize(request):    
    return render(request,'A_InputVisualize.html')
def N_InputVisualize(request):    
    return render(request,'N_InputVisualize.html')
def input_predict(request):    
    return render(request,'input_predict.html')
def solution(request):    
    return render(request,'solution.html')
def input_Compare(request):    
    return render(request,'input_Compare.html')



def pre(x,model_arima,model_prophet,pitem,years,p,d,q):
    df = pd.read_csv(x)
    df.drop(['Domain','Area','Element','Item'],axis=1,inplace=True)
    df['Year'] = df['Year'].astype(str) + '/12/31'
    df['Year'] = df['Year'].str.replace('/','-')
    df['Year'] = pd.to_datetime(df['Year'])
    df1=pd.DataFrame(df)
    df1.columns=['ds','y']
    #Facebook Prophet
    m = pickle.load(open(model_prophet, 'rb'))
    future = m.make_future_dataframe(periods=years,freq='Y',include_history=True)
    acc_future = m.make_future_dataframe(periods=0,freq='Y',include_history=True)
    acc_fut_p = m.predict(acc_future)
    mape_p,rmse_p = forecast_accuracy(acc_fut_p.yhat,df1.y)
    prophet = m.predict(future)
    prophet.drop(["trend","yhat_lower","yhat_upper","trend_lower","trend_upper","additive_terms","additive_terms_lower","additive_terms_upper","yearly","yearly_lower","yearly_upper","multiplicative_terms","multiplicative_terms_lower","multiplicative_terms_upper"],axis=1,inplace=True)
    #Neural Prophet
    df6 = pd.read_csv('carbonemission/combined_data_neural_prophet.csv',parse_dates=True)
    df9 = pd.DataFrame(df6.loc[df6['Item'] == pitem])
    df7 =  pd.DataFrame(df9[:57+years])
    #df7['ds'] = pd.to_datetime(df7['ds'])
    df8 =  pd.DataFrame(df6[:57])
    df8.rename(columns = {'yhat1':'forecast'}, inplace = True)
    mape_np,rmse_np = forecast_accuracy(df8.forecast,df1.y)
    #Arima
    df.columns=['Year','Value']
    model_fit=pickle.load(open(model_arima,'rb'))
    forecast = model_fit.predict(start=1,end = len(df.Value)+years,typ = 'levels').rename('Forecast')
    acc_fut_a = model_fit.predict(start=1,end = len(df.Value),typ = 'levels').rename('Forecast')
    mape_a,rmse_a = forecast_accuracy(acc_fut_a,df.Value)
    acc_list=[mape_p,rmse_p,mape_a,rmse_a,mape_np,rmse_np]
    buff1 = [] 
    for x in range(0,57+years):
        buff1.append(x)
    buff = np.array(buff1)
    forecast.index=buff1  

    buf1 = [] 
    for x in range(1961, 2018+years):
        buf1.append(int(x))
    buf = np.array(buf1)
    
    # providing an index
    ser = pd.Series(buf)
    df = pd.DataFrame({'Year':ser, 'Value':forecast})
    df['Year'] = df['Year'].astype(str) + '/12/31'
    df['Year'] = df['Year'].str.replace('/','-')
    df['Year'] = pd.to_datetime(df['Year'])
    return df,prophet,df7,acc_list

def preprocessing(x,years,p,d,q):
    dffao = pd.read_csv(x)
    dffao.drop(['Domain','Area','Element','Item'],axis=1,inplace=True)
    dffao['Year'] = dffao['Year'].astype(str) + '/12/31'
    dffao['Year'] = dffao['Year'].str.replace('/','-')
    dffao['Year'] = pd.to_datetime(dffao['Year'])
    return dffao

def forecast_accuracy(forecast, actual):
    mape = np.mean(np.abs(forecast - actual)/np.abs(actual))  # MApe
    rmse = np.mean((forecast - actual)**2)**.5  # RMSE
    return mape,rmse

def getPredictions(x,years):
    if x == 'Cereals':
        final,prophet,model_neural,acc_list=pre('carbonemission/indiaDataset/cereals_excluding_rice_old.csv',"carbonemission/darima/cereals_arima.pkl","carbonemission/prophet/cereals_excluding_prophet.pkl",'Cereals Exc Rice',years,10,1,10)
        final1=pd.DataFrame(preprocessing('carbonemission/indiaDataset/cereals_excluding_rice_old.csv', years,10,1,10))
    elif x== 'Buffalo Meat':
        final,prophet,model_neural,acc_list=pre('carbonemission/indiaDataset/meat_buffalo_old.csv',"carbonemission/darima/meat_buffalo_arima.pkl","carbonemission/prophet/meat_buffalo_prophet.pkl",'Meat Buffalo', years,30,1,40)
        final1=pd.DataFrame(preprocessing('carbonemission/indiaDataset/meat_buffalo_old.csv', years,30,1,40))
    elif x== 'Beef Meat':
        final,prophet,model_neural,acc_list=pre('carbonemission/indiaDataset/meat_cattle_old.csv',"carbonemission/darima/meat_cattle_arima.pkl","carbonemission/prophet/meat_cattle_prophet.pkl",'Meat Cattle',years,30,1,35)
        final1=pd.DataFrame(preprocessing('carbonemission/indiaDataset/meat_cattle_old.csv', years,30,1,35))
    elif x== 'Chicken Meat':
       final,prophet,model_neural,acc_list=pre('carbonemission/indiaDataset/meat_chicken_old.csv',"carbonemission/darima/meat_chicken_arima.pkl","carbonemission/prophet/meat_chicken_prophet.pkl",'Meat Chicken', years,10,1,7)
       final1=pd.DataFrame( preprocessing('carbonemission/indiaDataset/meat_chicken_old.csv', years,10,1,7))
    elif x== 'Goat Meat':
        final,prophet,model_neural,acc_list=pre('carbonemission/indiaDataset/meat_goat_old.csv', "carbonemission/darima/meat_goat_arima.pkl","carbonemission/prophet/meat_goat_prophet.pkl",'Meat Goat',years,30,1,40)
        final1=pd.DataFrame(preprocessing('carbonemission/indiaDataset/meat_goat_old.csv', years,30,1,40))
    elif x== 'Pig Meat':
        final,prophet,model_neural,acc_list=pre('carbonemission/indiaDataset/meat_pig_old.csv',"carbonemission/darima/meat_pig_arima.pkl","carbonemission/prophet/meat_pig_prophet.pkl",'Meat Pig', years,30,1,20)
        final1=pd.DataFrame(preprocessing('carbonemission/indiaDataset/meat_pig_old.csv', years,30,1,20))
    elif x== 'Sheep Meat':
       final,prophet,model_neural,acc_list=pre('carbonemission/indiaDataset/meat_sheep_old.csv',"carbonemission/darima/meat_sheep_arima.pkl","carbonemission/prophet/meat_sheep_prophet.pkl",'Meat Sheep', years,20,1,20)
       final1=pd.DataFrame( preprocessing('carbonemission/indiaDataset/meat_sheep_old.csv', years,20,1,20))
    elif x== 'Cow Milk':
        final,prophet,model_neural,acc_list=pre('carbonemission/indiaDataset/milk_whole__fresh_cow_old.csv',"carbonemission/darima/milk_cow_arima.pkl","carbonemission/prophet/milk_whole_fresh_cow_prophet.pkl",'Milk Whole Fresh Cow',years,30,1,20)
        final1=pd.DataFrame(preprocessing('carbonemission/indiaDataset/milk_whole__fresh_cow_old.csv',years,30,1,20))
    elif x== 'Buffalo Milk':
        final,prophet,model_neural,acc_list=pre('carbonemission/indiaDataset/milk_whole_fresh_buffalo_old.csv',"carbonemission/darima/milk_buffalo_arima.pkl","carbonemission/prophet/milk_whole_fresh_buffalo_prophet.pkl",'Milk Whole Fresh Buffalo', years,30,1,20)
        final1=pd.DataFrame(preprocessing('carbonemission/indiaDataset/milk_whole_fresh_buffalo_old.csv', years,30,1,20))
    else:
        final,prophet,model_neural,acc_list=pre('carbonemission/indiaDataset/eggs_hen_in_shell_old.csv',"carbonemission/darima/egg_hen_arima.pkl","carbonemission/prophet/eggs_hen_in_shell_prophet.pkl",'Egg Hen in Shell', years,30,1,20)
        final1=pd.DataFrame(preprocessing('carbonemission/indiaDataset/eggs_hen_in_shell_old.csv', years,30,1,20))
    final=pd.DataFrame(final)
    prophet=pd.DataFrame(prophet)
    model_neural=pd.DataFrame(model_neural)
    return final,final1,prophet,model_neural,acc_list
    

def getVisiualization(start,end):

    combined_data=pd.read_csv('carbonemission/combined_data.csv')
    combined_data=combined_data.loc[(combined_data['Year']>=start)&(combined_data['Year']<=end)]
    fig1 = px.bar(combined_data, x="Year", y="Emission Intensity", color='Item',width=930, height=580)
    plot_div1 = plot(fig1, output_type='div', include_plotlyjs=False)
    fig2 = px.line(combined_data, x='Year', y='Emission Intensity', color='Item',width=930, height=580)
    plot_div2 = plot(fig2, output_type='div', include_plotlyjs=False)
    x = combined_data['Year']
    y = combined_data['Item']
    z = combined_data['Emission Intensity']
    fig3 = go.Figure(data=go.Heatmap(z=z, x=x, y=y, colorscale='rdylgn'))
    fig3.update_layout(title='Carbon Emission in Meat Industry', xaxis_nticks=36,height=560,width=930)
    plot_div3 = plot(fig3, output_type='div', include_plotlyjs=False)

    fig4 = px.scatter(combined_data, y="Emission Intensity", x="Year", color="Item",height=580,width=930)
    fig4.update_traces(marker_size=10)
    plot_div4 = plot(fig4, output_type='div', include_plotlyjs=False)

    return plot_div1,plot_div2,plot_div3,plot_div4
    
   

def visualize(request):
    start =(request.GET['start_date'])
    end =(request.GET['end_date'])
    
    fig1,fig2,fig3,fig4=getVisiualization(start,end)

    context={'fig1':fig1,'fig2':fig2,'fig3':fig3,'fig4':fig4}
    return render(request,"visualize.html",context)
    






def result(request):
    years = int(request.GET['years'])
    items = request.GET['typeOfItem']
    # result,actual =  pd.DataFrame(getPredictions(items,years))
    result,actual,prophet,model_neural,acc =  getPredictions(items,years)
    result=pd.DataFrame(result)
    actual=pd.DataFrame(actual)
    #img = result.plot(x='ds',y='yhat')
    img = plot({'data':[Scatter(x=result['Year'], y=result['Value'],mode='lines+markers', name='Predicted Data', opacity=0.8, marker_color='blue'),Scatter(x=actual['Year'], y=actual['Value'],mode='lines+markers', name='Actual Data', opacity=0.8, marker_color='red')],'layout': {'xaxis': {'title': 'Year'}, 'yaxis': {'title': 'Carbon Emission'},'height':600}}, output_type='div')
    # plot({
    # 'data': [Scatter(...)],
    # 'layout': {'title': 'title', 'xaxis': {'title': 'xaxis_title'}, 'yaxis': {'title': 'yaxis_title'}}
    # }, output_type='div')
    # result=result.to_html(classes='table table-striped')
    # #print(result)
    # return render(request, 'result.html', {'result': result,'img':img})
    
    result['Year'] = result['Year'].dt.strftime('%Y-%m-%d')
    result['actualdata']=actual['Value']
    json_records = result.reset_index().to_json(orient ='records')
    data = []
    data = json.loads(json_records)
    context = {'d': data,'img':img,'items':items}
    return render(request, 'result.html', context)
    
    #return HttpResponse(result.to_html())
    # return render(request,'result.html',{'result':result})  


def com_res(request):
    years = int(request.GET['years'])
    items = request.GET['typeOfItem']
    # result,actual =  pd.DataFrame(getPredictions(items,years))
    result,actual,prophet,model_neural,acc_list =  getPredictions(items,years)
    result=pd.DataFrame(result)
    actual=pd.DataFrame(actual)
    prophet=pd.DataFrame(prophet)
    model_neural=pd.DataFrame(model_neural)
    img = plot({'data':[Scatter(x=result['Year'], y=result['Value'],mode='lines+markers', name='Arima Predicted Data', opacity=0.8, marker_color='black'),Scatter(x=actual['Year'], y=actual['Value'],mode='lines+markers', name='Actual Data', opacity=0.8, marker_color='red'),Scatter(x=model_neural['ds'], y=model_neural['yhat1'],mode='lines+markers', name='Neural Prophet Predicted Data', opacity=0.8, marker_color='yellow'),Scatter(x=prophet['ds'], y=prophet['yhat'],mode='lines+markers', name='Facebook Prophet Predicted Data', opacity=0.8, marker_color='blue')],'layout': {'xaxis': {'title': 'Year'}, 'yaxis': {'title': 'Emission Intensity'},'margin':{'t':15,'b':15},'height':650}}, output_type='div')
    
    result['Year'] = result['Year'].dt.strftime('%Y-%m-%d')
    result['actualdata']=actual['Value']
    result['prophet']=prophet['yhat']
    # result['neuralprophet']=model_neural['yhat1']
    json_records = result.reset_index().to_json(orient ='records')
    data = []
    data = json.loads(json_records)
    context = {'d': data,'img':img,'items':items,'acc_list':acc_list}
    return render(request, 'com_res.html', context)

# ARIMA visualize
def A_getVisiualization(start,end):

    combined_data=pd.read_csv('carbonemission/combined_data_arima.csv')
    combined_data=combined_data.loc[(combined_data['Year']>=start)&(combined_data['Year']<=end)]
    fig1 = px.bar(combined_data, x="Year", y="Emission Intensity", color='Item',width=930, height=580)
    plot_div1 = plot(fig1, output_type='div', include_plotlyjs=False)
    fig2 = px.line(combined_data, x='Year', y='Emission Intensity', color='Item',width=930, height=580)
    plot_div2 = plot(fig2, output_type='div', include_plotlyjs=False)
    x = combined_data['Year']
    y = combined_data['Item']
    z = combined_data['Emission Intensity']
    fig3 = go.Figure(data=go.Heatmap(z=z, x=x, y=y, colorscale='rdylgn'))
    fig3.update_layout(title='Carbon Emission in Meat Industry', xaxis_nticks=36,height=560,width=930)
    plot_div3 = plot(fig3, output_type='div', include_plotlyjs=False)

    fig4 = px.scatter(combined_data, y="Emission Intensity", x="Year", color="Item",height=580,width=930)
    fig4.update_traces(marker_size=10)
    plot_div4 = plot(fig4, output_type='div', include_plotlyjs=False)

    return plot_div1,plot_div2,plot_div3,plot_div4
    
   

def A_visualize(request):
    start =(request.GET['start_date'])
    end =(request.GET['end_date'])
    
    fig1,fig2,fig3,fig4=A_getVisiualization(start,end)

    context={'fig1':fig1,'fig2':fig2,'fig3':fig3,'fig4':fig4}
    return render(request,"A_visualize.html",context)

# NeuralProphet visualize
def N_getVisiualization(start,end):

    combined_data=pd.read_csv('carbonemission/combined_data_neural_prophet.csv')
    combined_data.rename(columns = {'ds' : 'Year', 'yhat1' : 'Emission Intensity'}, inplace = True)
    combined_data=combined_data.loc[(combined_data['Year']>=start)&(combined_data['Year']<=end)]
    fig1 = px.bar(combined_data, x="Year", y="Emission Intensity", color='Item',width=930, height=580)
    plot_div1 = plot(fig1, output_type='div', include_plotlyjs=False)
    fig2 = px.line(combined_data, x='Year', y='Emission Intensity', color='Item',width=930, height=580)
    plot_div2 = plot(fig2, output_type='div', include_plotlyjs=False)
    x = combined_data['Year']
    y = combined_data['Item']
    z = combined_data['Emission Intensity']
    fig3 = go.Figure(data=go.Heatmap(z=z, x=x, y=y, colorscale='rdylgn'))
    fig3.update_layout(title='Carbon Emission in Meat Industry', xaxis_nticks=36,height=560,width=930)
    plot_div3 = plot(fig3, output_type='div', include_plotlyjs=False)

    fig4 = px.scatter(combined_data, y="Emission Intensity", x="Year", color="Item",height=580,width=930)
    fig4.update_traces(marker_size=10)
    plot_div4 = plot(fig4, output_type='div', include_plotlyjs=False)

    return plot_div1,plot_div2,plot_div3,plot_div4
    
    
   

def N_visualize(request):
    start =(request.GET['start_date'])
    end =(request.GET['end_date'])
    
    fig1,fig2,fig3,fig4=N_getVisiualization(start,end)

    context={'fig1':fig1,'fig2':fig2,'fig3':fig3,'fig4':fig4}
    return render(request,"N_visualize.html",context)
