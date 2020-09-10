import argparse
import itertools
import pandas as pd
import numpy as np
import torch
import csv
import plotly.express as px
import chart_studio
import chart_studio.plotly as py
import chart_studio.tools as tls
from sklearn.manifold import TSNE

def parse_arguments():
    parser = argparse.ArgumentParser(description="NLProt Plot")
    parser.add_argument("papers", default=False)
    parser.add_argument("embeddings", default=False)
    parser.add_argument("apikey", default=False)
    
    return parser.parse_args()

def plot_tsne(papers, embeddings):
    emb = torch.load(embeddings)
    out = TSNE(n_components=2, perplexity=8, random_state=42).fit_transform(emb.numpy())

    with open(papers, "r") as f:
        abstracts = list(csv.DictReader(f))
    
    # create interactive plot
    df = pd.DataFrame(data=out, columns=['tSNE-1', 'tSNE-2'])
    titles = [abstract['title'] for abstract in abstracts]
    cap1, cap2 = 108, 107
    aut = [abstract['authors'] for abstract in abstracts]
    for i in range(len(aut)):
        if len(aut[i]) > cap1:
            aut[i] = aut[i][:cap1]+'...'
        if len(titles[i]) > cap2:
            titles[i] = titles[i][:cap2]+'...'
    df['authors'] = aut
    df['titles'] = titles
    date = [i['date'] for i in abstracts]
    years = [i['date'].split()[1] for i in abstracts]

    months = ['January','February','March','April','May','June',
              'July','August','September','October','November','December']
    yrs = ['2015','2016','2017','2018','2019','2020']

    
    x = list(itertools.product(yrs,months))
    x = [' '.join(x[i]) for i in range(len(x))]
    x = [x[i].split()[1] + ' ' + x[i].split()[0] for i in range(len(x))]
    sizes = np.linspace(2, 7, 72)
    dictionary = dict(zip(x, sizes))


    sizes = [dictionary[d]*7 for d in date]
    df['year'] = years
    df['authors'] = df['authors'].apply(lambda x: x.replace('|',', '))
    df['journal'] = [journal['journal'] for journal in abstracts]
    fig = px.scatter(df, x="tSNE-1", y="tSNE-2", template='simple_white', hover_name="titles", 
                     hover_data={'authors':True, 'tSNE-1': False, 'tSNE-2': False}, width=720, height=620)
    fig.update_traces(marker=dict(size=sizes, color='rgba(135, 206, 250, 0.9)', line=dict(width=1, color='white')), text=df['titles'], 
                      customdata=df['authors'],
                      hovertemplate=["<b>%{text}</b><br>" + "%{customdata}" + "<br>{0} | <i>{1}</i>".format(df['year'][i], df['journal'][i]) for i in range(len(df))])
    fig.update_xaxes(showticklabels=False, ticks='', visible=False)
    fig.update_yaxes(showticklabels=False, ticks='', visible=False)
    fig.update_layout({'plot_bgcolor': 'rgba(135, 206, 250, 0.07)',
                       'paper_bgcolor': 'rgba(250, 250, 250, 0)'
        })
    fig.update_layout(hovermode="closest", hoverlabel = dict(bgcolor='#3776ab', bordercolor='white', namelength=-1))
    fig.show(config=dict(displayModeBar=False))
    
    return fig
    
def api(key, fig): 
    username = 'tadorfer' 
    api_key = str(key)
    chart_studio.tools.set_credentials_file(username=username, api_key=api_key)
    py.plot(fig, filename='nlp_protein_papers', auto_open=False)
    print(tls.get_embed('https://plotly.com/~tadorfer/1/')) #change to your url

if __name__ == '__main__':
    args = parse_arguments()
    fig = plot_tsne(args.papers, args.embeddings)
    api(args.apikey, fig)