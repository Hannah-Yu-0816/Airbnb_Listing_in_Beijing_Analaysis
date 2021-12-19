from flask import Flask, render_template, request, jsonify
import numpy as np
import pandas as pd
import json
import plotly
import plotly.express as px
import plotly.graph_objects as go
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
from sklearn import preprocessing
import warnings
warnings.filterwarnings('ignore')

# import data
data = pd.read_csv('data.csv')
df = data.drop(['neighbourhood_group_cleansed', 'bathrooms', 'calendar_updated', 'license'], axis=1)
neighbourhood = df['neighbourhood_cleansed'].unique()
df['neighbourhood'] = df['neighbourhood_cleansed']

# host type
df.loc[df['calculated_host_listings_count'] <= 5, 'host_type'] = 'Personal'
df.loc[(df['calculated_host_listings_count'] <= 20) & (df['calculated_host_listings_count'] > 5),'host_type'] = 'Small scale company'
df.loc[(df['calculated_host_listings_count'] <= 100) & (df['calculated_host_listings_count'] > 20),'host_type'] = 'Medium scale company'
df.loc[df['calculated_host_listings_count'] > 100, 'host_type'] = 'Large scale company'

host_type_df = pd.DataFrame(np.zeros((16 * 4, 3)), columns=['neighbourhood', 'host type', 'count'])
for i in range(len(df['neighbourhood'].unique())):
    name = df['neighbourhood'].unique()[i]
    tmp = df[df['neighbourhood'] == name]
    h = tmp['host_type'].value_counts()
    host_type_df.loc[(0 + i * 4):(3 + i * 4), 'neighbourhood'] = name
    if len(h.index)!=4:
        host_type_df.loc[(0 + i * 4):(3 + i * 4), 'host type'] = ['Personal', 'Small scale company', 'Medium scale company',
       'Large scale company']
        host_type_df.loc[(0 + i * 4):(2 + i * 4), 'count'] = h.tolist()
        host_type_df.loc[3 + i * 4, 'count'] = 0
    else:
        host_type_df.loc[(0 + i * 4):(3 + i * 4), 'host type'] = h.index
        host_type_df.loc[(0 + i * 4):(3 + i * 4), 'count'] = h.tolist()
host_type_df['count'] = host_type_df['count'].astype(int)

# availability
df.loc[df['availability_365'] <= 60, 'availability'] = 'Highly available'
df.loc[df['availability_365'] > 60, 'availability'] = 'Lowly available'

a_df = pd.DataFrame(np.zeros((16 * 2, 3)), columns=['neighbourhood', 'availability', 'count'])
for i in range(len(df['neighbourhood'].unique())):
    name = df['neighbourhood'].unique()[i]
    tmp = df[df['neighbourhood'] == name]
    r = tmp['availability'].value_counts()
    a_df.loc[(0 + i * 2):(1 + i * 2), 'neighbourhood'] = name
    a_df.loc[(0 + i * 2):(1 + i * 2), 'availability'] = r.index
    a_df.loc[(0 + i * 2):(1 + i * 2), 'count'] = r.tolist()
a_df['count'] = a_df['count'].astype(int)

# room type
room_type_df = pd.DataFrame(np.zeros((16 * 3, 3)), columns=['neighbourhood', 'room type', 'count'])
for i in range(len(df['neighbourhood'].unique())):
    name = df['neighbourhood'].unique()[i]
    tmp = df[df['neighbourhood'] == name]
    r = tmp['room_type'].value_counts()
    room_type_df.loc[(0 + i * 3):(2 + i * 3), 'neighbourhood'] = name
    room_type_df.loc[(0 + i * 3):(2 + i * 3), 'room type'] = r.index
    room_type_df.loc[(0 + i * 3):(2 + i * 3), 'count'] = r.tolist()
room_type_df['count'] = room_type_df['count'].astype(int)

# income
df['reviews_per_month'] = df['reviews_per_month'].fillna(value=0)
df['income'] = df['minimum_nights'] * df['reviews_per_month'] * df['availability_30'] * df['price']

# kmeans

def model1(k, predict):
    analyse_data = df[['neighbourhood', 'latitude', 'longitude', 'room_type', 'accommodates', 'minimum_nights', 'availability_30', 'host_type', 'price', 'description', 'amenities']]
    analyse_data['neighbourhood_group'] = 0
    for i in range(len(df['neighbourhood'].unique())):
        name = df['neighbourhood'].unique()[i]
        index = df.index[df['neighbourhood'] == name].tolist()
        analyse_data['neighbourhood_group'][index] = i + 1

    analyse_data['room_type_group'] = 0
    for i in range(len(df['room_type'].unique())):
        name = df['room_type'].unique()[i]
        index = df.index[df['room_type'] == name].tolist()
        analyse_data['room_type_group'][index] = i + 1

    analyse_data['host_type_group'] = 0
    for i in range(len(df['host_type'].unique())):
        name = df['host_type'].unique()[i]
        index = df.index[df['host_type'] == name].tolist()
        analyse_data['host_type_group'][index] = i + 1

    clf = KMeans(n_clusters=k, random_state=42)
    clf.fit(analyse_data.drop(['price', 'room_type', 'host_type', 'neighbourhood', 'description', 'amenities'], axis=1))
    labels = clf.labels_.tolist()
    analyse_data['label'] = labels

    label = clf.predict(predict)

    return(analyse_data, label)


# knn
def model2(k, predict):
    analyse_data = df[
        ['neighbourhood', 'latitude', 'longitude', 'room_type', 'accommodates', 'minimum_nights', 'availability_30',
         'host_type', 'price', 'description', 'amenities']]
    analyse_data['neighbourhood_group'] = 0
    for i in range(len(df['neighbourhood'].unique())):
        name = df['neighbourhood'].unique()[i]
        index = df.index[df['neighbourhood'] == name].tolist()
        analyse_data['neighbourhood_group'][index] = i + 1

    analyse_data['room_type_group'] = 0
    for i in range(len(df['room_type'].unique())):
        name = df['room_type'].unique()[i]
        index = df.index[df['room_type'] == name].tolist()
        analyse_data['room_type_group'][index] = i + 1

    analyse_data['host_type_group'] = 0
    for i in range(len(df['host_type'].unique())):
        name = df['host_type'].unique()[i]
        index = df.index[df['host_type'] == name].tolist()
        analyse_data['host_type_group'][index] = i + 1

    m_scaler = preprocessing.MinMaxScaler()
    d = analyse_data.drop(['room_type', 'host_type', 'neighbourhood', 'price', 'description', 'amenities'], axis=1)
    samples = m_scaler.fit_transform(d)

    neigh = NearestNeighbors(n_neighbors=k)
    neigh.fit(samples)
    min = d.min()
    max = d.max()

    def transform(new):
        return (new - min) / (max - min)

    index = neigh.kneighbors([transform(predict)])[1].tolist()
    nearest = analyse_data.iloc[index[0], ]
    return(nearest)



# Flask
app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/wholecity')
def wholecityinonepage():
    return render_template('wholecityinonepage.html', graphJSON0=gm0(df), graphJSON1=gm1(df),
                           graphJSON2=gm2(room_type_df), graphJSON3=gm3(df), graphJSON4=gm4(host_type_df),
                           graphJSON5=gm5(df), graphJSON6=gm6(df), graphJSON7=gm7(a_df),
                           graphJSON8=gm8(df))

# listings distribution
def gm0(df):
    fig = go.Figure(data=[go.Pie(
        labels=df.neighbourhood.value_counts().index,
        values=df.neighbourhood.value_counts().tolist(),
        hole=.3)]
    )
    fig.update_layout(title_text="Airbnb Listings Distribution of Beijing")
    graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
    return graphJSON


# room type
def gm1(df):
    d = pd.DataFrame({
        "Room Type": df['room_type'].value_counts().index,
        "Counts": df['room_type'].value_counts().tolist()
    })
    fig = px.bar(d, x="Room Type", y="Counts", title="Airbnb Room Type in Beijing")
    graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
    return graphJSON

def gm2(df):
    fig = px.bar(df, x="neighbourhood", y="count", color="room type", title="Room Type in Each District")
    graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
    return graphJSON

# host type
def gm3(df):
    d = pd.DataFrame({
        "Host Type": df['host_type'].value_counts().index,
        "Counts": df['host_type'].value_counts().tolist()
    })
    fig = px.bar(d, x="Host Type", y="Counts", title="Airbnb Host Type in Beijing")
    graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
    return graphJSON

def gm4(df):
    fig = px.bar(df, x="neighbourhood", y="count", color="host type", title="Host Type in Each District")
    graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
    return graphJSON

# price
def gm5(df):
    fig = px.box(df, x="neighbourhood", y="price", notched=True)
    fig.update_layout(title_text="Price in Each District", yaxis=dict(range=[0, 7000]))
    graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
    return graphJSON

# availability
def gm6(df):
    d = pd.DataFrame({
        "Availability": df['availability'].value_counts().index,
        "Counts": df['availability'].value_counts().tolist()
    })
    fig = px.pie(d, names="Availability", values="Counts", title="Airbnb Room Availability of Beijing")
    graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
    return graphJSON

def gm7(df):
    fig = px.bar(df, x="neighbourhood", y="count", color="availability", title="Availability in Each District")
    graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
    return graphJSON

# income
def gm8(df):
    fig = px.box(df, x="neighbourhood", y="income", notched=True)
    fig.update_layout(title_text="Income of the host in Each District", yaxis=dict(range=[0, 40000]))
    graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
    return graphJSON


@app.route('/district', methods=["GET", "POST"])
def district():
    name = request.form.get("options")
    data = df[df['neighbourhood'] == name]
    return render_template('district.html', graphJSON9=gm9(name, data), graphJSON10=gm10(name, data),
                           graphJSON11=gm11(name, data), graphJSON12=gm12(name, data), graphJSON13=gm13(name, data))

# room type
def gm9(name, df):
    d = pd.DataFrame({
        "Room Type": df['room_type'].value_counts().index,
        "Counts": df['room_type'].value_counts().to_list()
    })
    fig = px.pie(d, names="Room Type", values="Counts", title=f'Room Type of {name}')
    graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
    return graphJSON

# host type
def gm10(name, df):
    d = pd.DataFrame({
        "Host Type": df['host_type'].value_counts().index,
        "Counts": df['host_type'].value_counts().to_list()
    })
    fig = px.pie(d, names="Host Type", values="Counts", title=f'Host Type of {name}')
    graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
    return graphJSON

# price
def gm11(name, df):
    fig = go.Figure()
    fig.add_trace(go.Box(x=df['price']))
    #fig = px.box(df, x="neighbourhood", y="income", notched=True)
    fig.update_layout(title_text=f"Price of the listings of {name}", xaxis=dict(range=[0, 5000]))
    graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
    return graphJSON

# availability
def gm12(name, df):
    d = pd.DataFrame({
        "Availability": df['availability'].value_counts().index,
        "Counts": df['availability'].value_counts().to_list()
    })
    fig = px.pie(d, names="Availability", values="Counts", title=f'Host Type of {name}')
    graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
    return graphJSON

# income
def gm13(name, df):
    fig = go.Figure()
    fig.add_trace(go.Box(x=df['income']))
    #fig = px.box(df, x="neighbourhood", y="income", notched=True)
    fig.update_layout(title_text=f"Income of the host of {name}", xaxis=dict(range=[0, 10000]))
    graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
    return graphJSON

@app.route('/knn', methods=["GET"])
def knn():
    k = int(request.args.get("input"))
    neighbourhood = int(request.args.get("neighbourhood"))
    latitude = float(request.args.get("latitude"))
    longitude = float(request.args.get("longitude"))
    room_type = int(request.args.get("room_type"))
    host_type = int(request.args.get("host_type"))
    accommodates = int(request.args.get("accommodates"))
    minimum_nights = int(request.args.get("minimum_nights"))
    availability_30 = int(request.args.get("availability_30"))
    new = [neighbourhood, latitude, longitude, room_type, host_type, accommodates, minimum_nights, availability_30]
    d1 = {1: 'Dongcheng', 2:'Xicheng', 3:'Changping', 4:'Daxing', 5:'Fangshan', 6:'Huairou', 7:'Mentougou', 8:'Miyun', 9:'Pinggu', 10:'Yanqing',
    11: 'Chaoyang', 12: 'Fengtai', 13:'Haidian', 14:'Shunyi', 15:'Tongzhou', 16:'Shijingshan'}
    d2 = {1:'Entire home/apt', 2:'Private room', 3:'Shared room'}
    d3 = {1:'Small scale company', 2:'Personal', 3:'Medium scale company', 4:'Large scale company'}
    entered = [d1[neighbourhood], latitude, longitude, d2[room_type], d3[host_type], accommodates, minimum_nights, availability_30]

    '''
    #predict = [[new]]
    m = model1(k=k, predict=new)
    analyse_data = m[0]
    label = int(m[1])
    count = []
    for i in range(k):
        num = len(analyse_data[analyse_data['label']==i]['price'])
        count.append({i:num})
    #count = analyse_data['label'].value_counts().tolist()
    price = analyse_data[analyse_data['label']==int(label)]['price']
    min = price.describe().tolist()[4]
    max = price.describe().tolist()[6]
    avg = price.describe().tolist()[5]
    '''

    #k-nn
    m2 = model2(k=k, predict=new).drop(['neighbourhood_group', 'room_type_group', 'host_type_group'], axis=1)
    m2.insert(0, 'No.', range(1, len(m2)+1), allow_duplicates=False)
    knn_return = []
    price = []
    for i in range(m2.shape[0]):
        knn_return.append(m2.iloc[i,:])
        price.append(m2.iloc[i,:]['price'])
    describe = pd.DataFrame(price).describe().values


    return render_template('advice.html', new=entered, knn=knn_return, k=k, price=price, d=describe)
    #return render_template('advice.html', new=new, graphJSON14=gm14(analyse_data), results=label, avg=int(avg), min=int(min), max=int(max), count=count, knn=knn_return)


def gm14(df):
    fig = px.box(df, x="label", y="price", notched=True)
    fig.update_layout(title_text="Price in different cluster", yaxis=dict(range=[0, 10000]))
    graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
    return graphJSON


if __name__ == '__main__':
    app.run()
