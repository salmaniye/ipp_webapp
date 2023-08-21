# /usr/local/bin/python3 -m pip install

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import io
import altair as alt
from collections import Counter
import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS
import json
import pickle
from os import listdir

# set deafult layout to wide
st.set_page_config(layout="wide")
header = st.container()
graph_header = st.container()

# initializing containers
with st.sidebar:
	input_container, word_cloud = st.tabs(["Control Panel", "Word Cloud"])

# st.markdown("""
# <style>
# .bigger-font {
#     font-size:18px !important;
# }
# </style>
# """, unsafe_allow_html=True)

# dataset = st.container()
fig_container = st.container()
common = st.container()
# plot_all = st.container()

# functions
@st.cache_data
def call_dataset(filename):
	# load data
	df = pd.read_csv(filename)
	df = df[df["sentiment"].str.contains('ERROR') == False]
	return df

# ----- NUMBER OF TWEETS PER PARTY -----

@st.cache_data
def create_df_tag_size(df):
	# for creating df with number of tweets
	df_tag_size = df.groupby(['tag'], as_index=False).size()
	return df_tag_size

# ----- NUMBER OF TWEETS PER SENTIMENT -----
@st.cache_data
def create_df_sentiment_size(df):
	# for creating df with number of tweets
	df_sentiment_size = df.groupby(['sentiment'], as_index=False).size()
	return df_sentiment_size

# ----- DATE RANGE ------

# @st.experimental_memo
# def func_slider_df_size(df,date_range):
# 	# for filtering dates of df with number of tweets
# 	df = df[df['datetime'].between(date_range[0],date_range[1],inclusive='both')]
# 	return df

# @st.experimental_memo
# def func_slider_df_all(df,date_range):
# 	# for filtering dates of df with original columns
# 	df = df[df['datetime'].between(date_range[0],date_range[1],inclusive='both')]
# 	return df

# ----- FILTERING DATAFRAME BY PARTY AND SENTIMENT -----
@st.cache_data
def function_filter_df(df,options_sentiment,options_party):
	# for filtering sentiment
	df = df[df["sentiment"].isin(options_sentiment)]
	df = df[df["tag"].isin(options_party)]
	return df

# ----- FILTERING BY KEYWORD -----
@st.cache_data
def search_keyword(df,key):
	# for filtering with text search
	df =  df[df['text'].str.contains(pat=key, case=False)==True]
	return df

# ----- HEADER -----
with header:
	st.markdown(f"## Sentiment of tweets on Indonesian Political Parties")
	st.markdown("""This is a web app that displays tweets and their sentiment on the selected Day.""")

# ----- GRAPH HEADER -----
with graph_header:
	fig_list = ['All parties','Gerindra','Golkar','NasDem','PAN','PDI-P',"PKB","PKS","PPP","PartaiDemokrat"]
	fig_pkl_list = ["fig_all.pkl","fig_gerindra.pkl","fig_golkar.pkl","fig_nasdem.pkl","fig_pan.pkl","fig_pdip.pkl","fig_pkb.pkl","fig_pks.pkl","fig_ppp.pkl","fig_partaidemokrat.pkl"]
	fig_dict =dict(zip(fig_list,fig_pkl_list))
	selected_fig = st.selectbox('Select party to display:',fig_list)

	fig_to_call = fig_dict.get(selected_fig)
	called_fig = pickle.load(open(fig_to_call,'rb'))
	called_fig.update_layout(width=1400)
	st.plotly_chart(called_fig)


# ----- CHOOSING CSV FILE -----

# function to remove prefix and suffix

def remove_both_sides(file_name):
	file_name = file_name.lstrip('sentiment_')
	file_name = file_name.rstrip('.csv')
	return file_name

with input_container:
	# a dropdown for the user to choose the game to be displayed
	filenames = sorted(listdir('sentiment_analysis/'))
	filelist = [filename for filename in filenames if filename.endswith(".csv")]

	day_date = st.selectbox('Select a day to display:', filelist, format_func=lambda x: remove_both_sides(x))
	st.caption(f'You have selected: {day_date}')

# ----- CREATING PARTY BAR CHART -----
@st.cache_data
def create_party_figure(df):
	# creates bar chart of number of tweets and party
	fig = px.bar(df, x='tag', y='size', color='tag')
	fig.update_layout(title_text=f"Number of tweets per party",
		font=dict(size=16))
	return fig

# ----- CREATING SENTIMENT BAR CHART -----
@st.cache_data
def create_sentiment_figure(df):
	# creates bar chart of number of tweets and sentiment
	fig = px.bar(df, x='sentiment', y='size', color='sentiment',
		color_discrete_map={'positive':'#109618','neutral':'#3366CC','negative':'#DC3912'}) 
		#['green', 'blue', 'red']
	fig.update_layout(title_text=f"Number of tweets per sentiment",
		font=dict(size=16))
	return fig

# @st.experimental_memo
# def func_creating_fig2(df):
# 	# creates plot of normalized sentiment with percentage
# 	fig2 = px.area(df, x='datetime', y='sentiment percentage',labels={
# 		'datetime':'Date',
# 		'sentiment percentage':'Sentiment (%)',
# 		'sentiment':'Sentiment'},
# 		color='sentiment',
# 		color_discrete_map={'Positive':'#109618','Neutral':'#3366CC','Negative':'#DC3912'},
# 		category_orders={"sentiment": ["Negative", "Neutral", "Positive"]})
# 	fig2.update_layout(title_text=f"Normalized sentiment of tweets over time", title_x=0.5,
# 		font=dict(size=16))
# 	return fig2



# ----- CREATE DATAFRAME -----
df = call_dataset("sentiment_analysis/"+day_date)

# ----- FILTERING OPTIONS CONTAINER -----
with input_container:
	inputs = st.form(key='form',clear_on_submit=False)

# # grouping sentiment per date
# sentiment_per_day = func_sentiment_per_day(game_dataset)
# min_date = sentiment_per_day['datetime'].min()
# max_date = sentiment_per_day['datetime'].max()

# date_range = list([0,0])

# ----- RESET OPTIONS -----
# function for clearing inputs by restting session states
def clear_inputs():
	#st.session_state['dateinput1'] = min_date
	#st.session_state['dateinput2'] = max_date
	st.session_state['opsentiment'] = ['positive', 'neutral', 'negative']
	st.session_state['kw_s'] = ""
	st.session_state['opparty'] = ['Gerindra','Golkar','NasDem','PAN','PDI-P',"PKB","PKS","PPP","PartaiDemokrat"]
	return

##############################################################################################
# ----- FILTERING -----
with inputs:
# 	# start and end dates
# 	date_range[0] = st.date_input('Select starting date:', min_date, min_date, max_date,key='dateinput1')
# 	date_range[1] = st.date_input('Select end date:', max_date, min_date, max_date,key='dateinput2')

# 	date_range[0] = pd.to_datetime(date_range[0])
# 	date_range[1] = pd.to_datetime(date_range[1])

	# options for filtering sentiment
	options_sentiment = st.multiselect(label='Filter by sentiment:',
		options=['positive', 'neutral', 'negative'],
		default=['positive', 'neutral', 'negative'],
		key='opsentiment')
	
	options_party = st.multiselect(label='Filter by party:',
				options=['Gerindra','Golkar','NasDem','PAN','PDI-P',"PKB","PKS","PPP","PartaiDemokrat"],
				default=['Gerindra','Golkar','NasDem','PAN','PDI-P',"PKB","PKS","PPP","PartaiDemokrat"],
				key='opparty')

	# search text in dataframe
	keyword_text = st.text_input('Search text within the date range (case insensitive):', key='kw_s')
	if keyword_text:
		st.caption(f'The current text search is: {keyword_text}')
	else:
		st.caption(f'No text search input')

	# submit button
	submitted = st.form_submit_button("Click to Submit")

	if submitted:
		st.write("Submitted")
	#create your button to clear the state of the multiselect

with input_container:
	st.button("Reset options to default values", on_click=clear_inputs)

##############################################################################################

if keyword_text:
	df = search_keyword(df,keyword_text)

# # dataframe for number of tweets
# sentiment_per_day = func_sentiment_per_day(game_dataset)
# slider_df = func_slider_df_size(sentiment_per_day,date_range)
# slider_df = slider_df[slider_df["sentiment"].isin(options_sentiment)]

# creates a dataframe of tweets created between dates chosen
# date_range_df = func_slider_df_all(game_dataset,date_range)
filtered_df = function_filter_df(df,options_sentiment,options_party)
if keyword_text:
	filtered_df = search_keyword(filtered_df,keyword_text)

# # fig1. sentiment over time
# fig = func_creating_fig1(slider_df)

# # fig2. normalized sentiment area over time
# @st.experimental_memo
# def func_spd(df):
# 	sentiment_total_pd = df.groupby(['datetime'], as_index=False).sum()
# 	spd = df.merge(sentiment_total_pd, left_on = 'datetime', right_on='datetime')
# 	spd['sentiment percentage'] = 100*(spd['size_x']/spd['size_y'])
# 	return spd

# fig2 = func_creating_fig2(func_spd(slider_df))

total_number_of_tweets = len(filtered_df['text'])
positive_percentage = 100*len(filtered_df[filtered_df['sentiment']=='positive'])/len(filtered_df['sentiment'])
neutral_percentage = 100*len(filtered_df[filtered_df['sentiment']=='neutral'])/len(filtered_df['sentiment'])
negative_percentage = 100*len(filtered_df[filtered_df['sentiment']=='negative'])/len(filtered_df['sentiment'])

df_tag_size = create_df_tag_size(filtered_df)
df_sentiment_size = create_df_sentiment_size(filtered_df)

fig_party_size = create_party_figure(df_tag_size)
fig_sentiment_size = create_sentiment_figure(df_sentiment_size)

with fig_container:
	st.write(fig_party_size)
	st.markdown(f"""<p class="bigger-font">
	Total Number of Tweets: <b>{total_number_of_tweets}</b><br>
	Positive: <b>{positive_percentage:.2f}%</b><br>
	Neutral: <b>{neutral_percentage:.2f}%</b><br>
	Negative: <b>{negative_percentage:.2f}%</b></p>""",
	unsafe_allow_html=True)
	st.write(fig_sentiment_size)

# ------- WORD CLOUD --------

def wordcloud_generator():
	dataset_text = ' '.join(filtered_df['text']) # edit this

	# with open(f"metadata/custom_stopwords.txt","r") as file: # edit this
	# 	custom_stopwords = []
	# 	for line in file:
	# 		line = line.rstrip("\n")
	# 		custom_stopwords.append(line)

	# stopwords_all = custom_stopwords + list(STOPWORDS)

	stopwords_all = list(STOPWORDS) + ['user']
	fig_word, ax = plt.subplots()

	wordcloud = WordCloud(background_color='white', colormap='Set2',
				width = 1000, height=2600,
				collocations=False, stopwords = stopwords_all).generate(dataset_text)

	plt.imshow(wordcloud, interpolation='bilinear')
	plt.axis("off")
	plt.show()
	return fig_word

with word_cloud:

	# st.markdown('Word cloud of most common words between the date range and text search')
	# st.caption('The wordcloud is used to inspire you to find words to use in the text search in the Control Panel')
	fig_word = wordcloud_generator()
	st.pyplot(fig_word)

with common:
	st.dataframe(filtered_df)
