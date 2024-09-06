import requests
import json
import time
import csv
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from tqdm import tqdm

API_KEY = '2f5f2fc6c4e8674dbfc7bb1a7cd2a52d'  # Replace with your TMDB API key

def make_request(url):
    retries = 7
    for i in range(retries):
        try:
            response = requests.get(url, timeout=10)  # Set a timeout value
            if response.status_code == 200:
                return response.json()
            elif response.status_code == 429:  # Rate limit exceeded
                retry_after = int(response.headers.get("Retry-After", 2))
                time.sleep(retry_after)
            else:
                response.raise_for_status()  # Raise an error for other bad status codes
        except requests.exceptions.Timeout:
            print(f"Timeout occurred for URL: {url}. Retrying ({i + 1}/{retries})...")
            time.sleep(2 ** i)  # Exponential backoff
        except requests.exceptions.RequestException as e:
            print(f"Request failed for URL: {url}. Error: {e}")
            break
    return None

def get_movies_by_year(year):
    movies = []
    for page in tqdm(range(1, 501), desc=f"Fetching movies for {year}"):
        url = f"https://api.themoviedb.org/3/discover/movie?api_key={API_KEY}&language=en-US&sort_by=popularity.desc&include_adult=false&page={page}&primary_release_year={year}"
        data = make_request(url)
        
        if data is None:
            continue
        
        for movie in data['results']:
            if movie['original_language'] == 'en':
                movies.append(movie['id'])
        
        #time.sleep(0.2)  # To avoid hitting the rate limit
    return movies

def get_movie_details(movie_id):
    url = f"https://api.themoviedb.org/3/movie/{movie_id}?api_key={API_KEY}&language=en-US"
    return make_request(url)

def get_director(movie_id):
    url = f"https://api.themoviedb.org/3/movie/{movie_id}/credits?api_key={API_KEY}&language=en-US"
    credits = make_request(url)
    if credits is None:
        return None
    director = next((person for person in credits['crew'] if person['job'] == 'Director'), None)
    return director['id'] if director else None

def get_person_popularity(person_id):
    url = f"https://api.themoviedb.org/3/person/{person_id}?api_key={API_KEY}&language=en-US"
    person = make_request(url)
    return person['popularity'] if person else None

def get_top_cast_popularity(movie_id):
    url = f"https://api.themoviedb.org/3/movie/{movie_id}/credits?api_key={API_KEY}&language=en-US"
    credits = make_request(url)
    if credits is None:
        return None
    cast = credits['cast']
    return sum(member['popularity'] for member in cast)

def get_movie_reviews(movie_id):
    url = f"https://api.themoviedb.org/3/movie/{movie_id}/reviews?api_key={API_KEY}&language=en-US"
    reviews_data = make_request(url)
    if not reviews_data['results']:
        return []
    return [review['content'] for review in reviews_data['results']]

def analyze_sentiment(reviews):
    if not reviews:
        return "NaN"
    sentiment_scores = []
    sia = SentimentIntensityAnalyzer()
    for review in reviews:
        sentiment_scores.append(sia.polarity_scores(review)['compound'])
    return sum(sentiment_scores) / len(sentiment_scores)

def main():
    movie_data = []
    for year in range(2024, 2025):  # Adjust the range as needed
        movie_ids = get_movies_by_year(year)
        
        for movie_id in tqdm(movie_ids, desc=f"Processing movies for {year}"):
            movie_details = get_movie_details(movie_id)
            
            if movie_details is None or movie_details['vote_count'] == 0 or movie_details['budget'] == 0:
                continue
            
            director_id = get_director(movie_id)
            director_popularity = get_person_popularity(director_id) if director_id else None
            cast_popularity = get_top_cast_popularity(movie_id)
            reviews = get_movie_reviews(movie_id)
            sentiment_score = analyze_sentiment(reviews)
            
            movie_info = {
                "title": movie_details['title'],
                "vote_average": movie_details['vote_average'],
                "vote_count": movie_details['vote_count'],
                "director_popularity": director_popularity,
                "cast_popularity": cast_popularity,
                "genres": [genre['name'] for genre in movie_details['genres']],
                "movie_popularity": movie_details['popularity'],
                "production_companies": [company['name'] for company in movie_details['production_companies']],
                "budget": movie_details['budget'],
                "revenue": movie_details['revenue'],
                "runtime": movie_details['runtime'],
                "release_date_month": movie_details['release_date'][:7] if movie_details['release_date'] else None,
                "tmdb_id": movie_details['id'],
                "imdb_id": movie_details['imdb_id'],
                "sentiment": sentiment_score
            }
            
            movie_data.append(movie_info)
            #time.sleep(0.2)  # To avoid hitting the rate limit

    # Save data to CSV file
    with open('tmdb_movie_data2.csv', 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=movie_data[0].keys())
        writer.writeheader()
        writer.writerows(movie_data)

if __name__ == "__main__":
    main()
