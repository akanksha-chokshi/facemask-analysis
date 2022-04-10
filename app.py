import pandas as pd
import numpy as np
from sklearn import cluster
import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import altair as alt
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
import ssl
ssl._create_default_https_context = ssl._create_unverified_context
from rake_nltk import Rake
rake_nltk_var = Rake()
sia = SentimentIntensityAnalyzer()
scalar = StandardScaler()

from PIL import Image
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
import matplotlib.pyplot as plt

@st.cache (persist = True)
def read_files():
    nltk.download('vader_lexicon')
    nltk.download('stopwords')
    products=pd.read_csv('products.csv')
    reviews=pd.read_csv('reviews.tsv',sep='\t')
    products.set_index('product_id', inplace=True)
    products['brand'] = products['product_name'].str.split(',').str[0]
    products['brand'].str.replace(',', '')  
    products = products[products.product_price != 49.61]
    reviews.rename(columns={"productId": "product_id"}, inplace=True)
    reviews["reviews"] = np.where(reviews["languageCode.1"] == "en-US", reviews["reviewText"], reviews["translation.reviewText"])
    merged = reviews.join(products, on='product_id')
    merged['sentiment_score'] = merged['reviews'].apply(lambda x: sia.polarity_scores(x)['compound'])
    merged['sentiment'] = merged['sentiment_score'].apply(lambda x: sentiment_classify(x))
    return products, reviews, merged

@st.cache (persist = True)
def queries (products, reviews, merged):
    brand_products = products.groupby('brand')['product_name'].count().sort_values(ascending=False).to_frame(name = "Number of Products Listed")
    products_grouped = merged.groupby('product_name')[['ratingValue', 'quantity', 'price', 'product_price']].mean().sort_values(by = "ratingValue", ascending=False)
    brands_grouped = merged.groupby('brand')[['ratingValue', 'quantity', 'price', 'product_price']].mean().sort_values(by = "ratingValue", ascending=False)
    types_grouped = merged.groupby('type')[['ratingValue', 'quantity', 'price', 'product_price']].mean().sort_values(by = "ratingValue", ascending=False)
    products_encoded = pd.get_dummies(products['type'])
    products_encoded['product_name'] = products['product_name'].astype(str)
    products_encoded = products_encoded.set_index('product_name')
    products_merged = products_grouped.join(products_encoded)
    p_copy = products_merged.copy()
    p_copy = scalar.fit_transform(p_copy).astype(float)
    kmeans = KMeans(n_clusters=5, random_state=45).fit(products_merged)
    labels = kmeans.labels_
    products_merged['labels'] = labels
    cluster_ratings = products_merged.groupby('labels')[['ratingValue', 'quantity', 'price','product_price']].mean()
    cluster_ratings['category'] = ["General", "Cost", "Health", "Comfort", "Environment"]
    cluster_ratings = cluster_ratings.set_index(['category']).sort_values(by = "ratingValue", ascending = False)
    types_grouped_sentiment = merged.groupby('type')[['sentiment_score', 'ratingValue']].mean()
    p_grouped = merged.groupby('product_name')[['sentiment_score','ratingValue', 'quantity', 'price']].mean().sort_values(by = "ratingValue", ascending=False)
    p_grouped = p_grouped.join(products_encoded)
    p_grouped['labels'] = labels
    cluster_sentiment = p_grouped.groupby('labels')[['sentiment_score', 'ratingValue']].mean()
    cluster_sentiment['category'] = ["General", "Cost", "Health", "Comfort", "Environment"]
    cluster_sentiment = cluster_sentiment.set_index(['category']).sort_values(by = "ratingValue", ascending = False)
    product_sentiment = p_grouped.groupby('product_name')[['sentiment_score', 'ratingValue', 'price']].mean().sort_values(by = "sentiment_score", ascending=False)
    product_sentiment['score'] = product_sentiment ['sentiment_score'] * product_sentiment['ratingValue']
    scored_products = product_sentiment.sort_values(by = "score", ascending = False)
    scored_products = pd.concat([scored_products, p_grouped[['quantity', 'labels']]], axis = 1)
    scored_products ['category'] = scored_products['labels'].apply(lambda x: category_classify(x))
    merged_labels = merged.join(products_merged['labels'], on = "product_name")[['labels', 'reviews', 'sentiment']]
    merged_labels['keywords'] = merged_labels['reviews'].apply(lambda x: get_keywords (x))
    return brand_products, products_grouped, brands_grouped, types_grouped, products_encoded, products_merged, cluster_ratings, types_grouped_sentiment, cluster_sentiment, product_sentiment, scored_products, merged_labels
    
def sentiment_classify (num):
    sentiment = ""
    if num < 0:
        sentiment = "negative"
    elif num < 0.5:
        sentiment = "neutral"
    else:
        sentiment = "positive"
    return sentiment

def category_classify (num):
    category = ""
    if num == 0:
        category = "General"
    elif num == 1:
        category = "Cost"
    elif num == 2:
        category = "Health"
    elif num == 3:
        category = "Comfort"
    else:
        category = "Environment"
    return category

def get_keywords (text):
    rake_nltk_var.extract_keywords_from_text(text)
    keyword_extracted = rake_nltk_var.get_ranked_phrases()
    return str(keyword_extracted)

products, reviews, merged = read_files()
stopwords = nltk.corpus.stopwords.words('english')
brand_products, products_grouped, brands_grouped, types_grouped, products_encoded, products_merged, cluster_ratings, types_grouped_sentiment, cluster_sentiment, product_sentiment, scored_products, merged_labels = queries (products, reviews, merged)
st.title("Analysing the Facemasks Market")
st.subheader("Can we use iHerb consumer reviews to better understand facemasks consumers and their needs?")
st.markdown("Use the sidebar to navigate through the analysis.")
st.sidebar.title("Analysing the Facemasks Market")

@st.cache(persist=True)
def generate_wordcloud(merged_labels):
    text = " ".join(s for s in merged_labels.keywords.astype(str))
    text = text.replace("'", "")
    stopwords = set(STOPWORDS)
    stopwords.update(['mask', 'masks', 'face', 'wear', 'good', 'nice', 'well', 'great'])
    wordcloud = WordCloud(stopwords = stopwords, background_color = "white", colormap='winter').generate (text)
    return wordcloud

prelim = st.sidebar.button("Initial Explorations")
if prelim:
    st.subheader("Data Source:")
    st.write("Two datasets scraped from the iHerb website containing product info and product reviews.")
    st.caption("Things to note about iHerb:")
    st.caption("An e-commerce site that sells health-related products across Asia and the United States.")
    st.caption("Known for low prices and variety of products.")
    st.caption("Consumers can only review products they have purchased.")
    with st.expander("Preprocessing Steps:"):
        st.write("Products:")
        st.write("Columns Added: Quantity (number of masks in the product), Price (average price given quantity, different from actual product price), Type (disposable N95, disposable regular or reusable masks), Brand (extracted from product name)")
        st.write("For duplicate products, I kept the most recent product information.")
        st.write("Reviews:")
        st.write("I created a new column for Reviews to store all English reviews (combining the general reviews and translated columns).")
        st.write("I performed sentiment analysis on the Reviews and created two new columns - one for the sentiment score as a number and one to store the category.")
        st.write("Lastly, merged the two dataframes on product_id. The preprocessing as well as subsequent queries are cached within the Streamlit app to avoid computing them everytime the app is run.")
    st.subheader("How many Brands and Products are we working with?")
    st.metric ("Number of Brands", merged['brand'].nunique())
    st.metric ("Number of Products", merged['product_name'].nunique())
    st.bar_chart(brand_products)
    st.caption("Some brands are way more represented in the data than others.")
    st.write("Note: Reviews do not indicate sales or overall consumer perception.")


popular = st.sidebar.button("What Do The Ratings Say?")
if popular:
    st.subheader("Most Popular Products")
    st.caption("These are the 10 most popular products, given their average rating value.")
    st.table(products_grouped.head(10).index)
    with st.expander ("View Data"):
        st.write(products_grouped)
    st.subheader("Most Popular Brands")
    st.caption("These are the 5 most popular brands, given the average rating value of their products.")
    st.table(brands_grouped.head(5))
    st.subheader("Initial Observations")
    st.write("On looking at the ranked list of products and brands, we can see that most of the top 10-15 products come from the top 5 brands. Even when we look at all the products (based on their rank), they are heavily weighted towards the top 5 brands.")
    st.write("We can identify some traits from these popular products - they are all sold in smaller quantities (usually 1-3, with a few stark exceptions) and priced similarly between 2-6 AUD per mask.")
    st.write("However, I noticed a distinct difference in the nature of the masks they sell. For example, Zidian only sells disposable masks while Lozperi is known for its range of reusable cotton masks. Given that the top 5 brands focus on a mix of reusable, disposable (regular and N95 masks), I wanted to investigate these different types and see if they lead to different consumer personas.")


types = st.sidebar.button("Can We Create Personas?")
if types:
    st.subheader("Different Types of Facemasks and their Characteristics:")
    st.write(types_grouped)
    st.write("On looking at this table, I realised that while the overall product price of all three types is quite similar (12-18 AUD, with reusable masks on the more expensive end), disposable regular masks are sold in a package of 25, while the others are sold in average packs of 2-3 for that price.")
    st.write("Rating values are quite close to each other in general, but we can see that disposable regular masks are rated more favourably, followed by disposable N96 and then reusable.")
    st.write("However, I could see how each type of mask attracts a different type of consumer and the cheapest price is not always best.")
    st.write("To explore different consumer personas, I decided to cluster products by their rating, quantity and price, one-hot encoded by type, to see if we could pick up on underlying motivations behind their purchases.")
    st.subheader("Cluster 1: The General Consumers")
    with st.expander ("View Data"):
        st.write(products_merged[products_merged['labels'] == 0])
    st.caption("This cluster may not seem to imply significantly meaningful motivations - in contrast to the clusters we will see later on. However, at face value, it seems to be a well-balanced mix of prioritising cost, comfort and health at low quantities and moderate prices. A safe bet for brands/products that do not capitalise on a specific niche. A variety of rating values from extremely well-rated to reasonably lower ratings.")
    st.subheader("Cluster 2: Low Cost/High Convenience Seekers")
    with st.expander ("View Data"):
        st.write(products_merged[products_merged['labels'] == 1])
    st.caption("This cluster shows products that attract consumers who prioritise the lowest costs possible. They are the most affordable options and are sold in large quantities for the sake of convenience. Rating highly fluctuates in this case as well - and it seems that it is not a priority for consumers who are simply looking for the most cost-effective solution. All masks are disposable regulars - not high on comfort or health.")
    st.subheader("Cluster 3: The Health and Safety Enthusiasts")
    with st.expander ("View Data"):
        st.write(products_merged[products_merged['labels'] == 2])
    st.caption ("This cluster also only contains disposable masks - however, they are all N95s and relatively more expensive than the previous cluster. These consumers value the safety of the N95 mask and are willing to pay more for it. They also purchase these in moderately high quantities (20-25 a package) since they prefer the safety and security they promise.")
    st.subheader("Cluster 4: The Comfort Chasers")
    with st.expander ("View Data"):
        st.write(products_merged[products_merged['labels'] == 3])
    st.caption ("This cluster features only reusable and some disposable N95s which are priced relatively higher than the general reusable masks from the first cluster. Looking at the title, we can see they emphasise 'cotton' and 'fashion'able masks - showing they are of a premium quality and fit compared to others. These are also sold in relatively smaller quantities.")
    st.subheader("Cluster 5: The Outliers")
    with st.expander ("View Data"):
        st.write(products_merged[products_merged['labels'] == 4])
    st.caption("This cluster stands out because it has only one product - it is interesting that it does because I actually didn't know how to classify it. It is one mask with 24 reusable filters, which explains the stark difference in price when you treat it as 24 masks. I, however, decided to classify it as 24 masks since that is the value provided to the consumer. It also has additional environmental benefits due to reduced waste - so that is another relatively unexplored consumer profile that could be targeted.")
    st.subheader("Cluster Stats")
    st.write("How do these clusters perform in general across different categories?")
    with st.expander ("View Data"):
        st.write(cluster_ratings)
    st.write("Although we previously explored that cost could be the primary motivating factor behind ratings, we now see underlying motivations emerge and that cost is actually the least important motivator (apart from the outlier).")
    st.write("Those who prioritise health and safety tend to rate their masks the best, followed by the general consumers and then those who prefer comfort.")
    st.write("It's interesting to see the quantity and price fluctuations - those who appeal to the general/comfort-seeking consumers tend to sell masks in much smaller quantities compared to the others. Masks catered towards the comfort-seeking consumer are also significantly more expensive per mask. However, the largest one-time price is paid by the health enthusiasts who purchase a moderate quantity at a moderate price but are willing to do so out of commitment for their health.")

reviews = st.sidebar.button("What Do Their Reviews Say?")
if reviews:
    st.subheader("Analysing Review Sentiment")
    st.metric("Total Reviews", len(merged['sentiment']))
    st.metric("Positive Reviews", len(merged[merged['sentiment'] == "positive"]))
    st.metric ("Negative Reviews", len(merged[merged['sentiment'] == "negative"]))
    st.metric ("Neutral Reviews", len(merged[merged['sentiment'] == "neutral"]))
    with st.expander ("How did I classify these?"):
        st.write("I used NLTK's Vader_Lexicon model which classifies negative text below 0, neutral text at 0 and positive text above 0. However, I noticed that most of the reviews were scored above 0, with very few at or below 0. Hence, I adjusted the bounds of the classification a bit and made all reviews between 0 and 0.5 neutral, and only those above 0.5 as positive. We can still see the significant amount of positive ratings, which coupled with the relatively high rating values, might hint at a potential platform bias - only satisfied consumers are rating and reviewing (something to keep in mind).")
    st.subheader("How does this change for different types of facemasks?")
    types_grouped_sentiment = types_grouped_sentiment.reset_index()
    c = alt.Chart(types_grouped_sentiment).mark_bar().encode(alt.X('sentiment_score', scale = alt.Scale(domain = (0.45, 0.52))), y='type')
    st.altair_chart(c)
    c = alt.Chart(types_grouped_sentiment).mark_bar().encode(alt.X('ratingValue', scale = alt.Scale(domain = (42, 46))), y='type')
    st.altair_chart(c)
    with st.expander ("View Data"):
        st.write(types_grouped_sentiment)
    st.subheader("How does this change for different personas?")
    cluster_sentiment = cluster_sentiment.reset_index()
    c = alt.Chart(cluster_sentiment).mark_bar().encode(alt.X('sentiment_score', scale = alt.Scale(domain = (0.2, 0.6))), y='category')
    st.altair_chart(c)
    c = alt.Chart(cluster_sentiment).mark_bar().encode(alt.X('ratingValue', scale = alt.Scale(domain = (39, 46))), y='category')
    st.altair_chart(c)
    with st.expander ("View Data"):
        st.write(cluster_sentiment)
    st.subheader("How does this change for different products?")
    st.caption("These are the 10 most favourably reviewed products, given their average sentiment score.")
    st.table(product_sentiment['sentiment_score'].head(10))
    st.write("Compared to the Top 10 most well-rated products:")
    st.table(products_grouped['ratingValue'].head(10))
    st.subheader("What if we layer the two metrics?")
    st.write("Top 10 well-rated and well-reviewed products:")
    st.table(scored_products[['score','category', 'quantity', 'price']].head(10))
    st.header("What are the reviews about?")
    wordcloud = generate_wordcloud(merged_labels)
    fig, ax = plt.subplots()
    ax.imshow(wordcloud, interpolation = "bilinear")
    plt.axis("off")
    st.pyplot(fig)
    
conclusion = st.sidebar.button("Conclusion")
if conclusion:
    st.header("Conclusion")

