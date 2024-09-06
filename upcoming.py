import requests
import csv
from datetime import datetime

# TMDB API Key
api_key = '2f5f2fc6c4e8674dbfc7bb1a7cd2a52d'

# Function to get the popularity of the director and cast
def get_people_popularity(credits):
    director_popularity = 0
    cast_popularity = 0
    cast_count = 0
    
    for crew_member in credits['crew']:
        if crew_member['job'] == 'Director':
            director_popularity = crew_member['popularity']
            break
    
    for cast_member in credits['cast']:
        cast_popularity += cast_member['popularity']
        cast_count += 1
    
    if cast_count > 0:
        cast_popularity /= cast_count  # Average popularity of cast
    
    return director_popularity, cast_popularity

# Function to fetch and process movie data
def fetch_upcoming_movies():
    page = 1
    total_pages = 1
    processed_movies = set()  # Set to track processed movies

    with open('upcoming_movies_no_votes.csv', 'w', newline='', encoding='utf-8') as no_votes_file:
        no_votes_writer = csv.writer(no_votes_file)
        headers = ['Name', 'Vote Average', 'Vote Count', 'Director Popularity', 'Cast Popularity', 'Genres', 'Movie Popularity', 'Production Companies', 'Budget', 'Revenue', 'Runtime', 'Release Date Month', 'TMDB ID', 'IMDB ID']
        no_votes_writer.writerow(headers)
        
        while page <= total_pages:
            url = f'https://api.themoviedb.org/3/movie/upcoming?api_key={api_key}&language=en-US&page={page}'
            response = requests.get(url)
            data = response.json()
            total_pages = data['total_pages']  # Update total pages
            movies = data['results']
            
            for movie in movies:
                if movie['id'] not in processed_movies and movie['original_language'] == 'en':  # Filter English movies and check for duplicates
                    details_url = f'https://api.themoviedb.org/3/movie/{movie["id"]}?api_key={api_key}&append_to_response=credits'
                    details_response = requests.get(details_url)
                    details = details_response.json()

                    if not details.get('imdb_id'):  # Skip movies without an IMDb ID
                        continue
                    
                    processed_movies.add(movie['id'])
                    
                    try:
                        release_date_month = datetime.strptime(details['release_date'], '%Y-%m-%d').strftime('%Y-%m')
                    except ValueError:
                        release_date_month = 'Unknown'  # Handle missing or incorrect date formats

                    if 'vote_count' in details and details['vote_count'] == 0:  # Include only movies with no votes
                        name = details['title']
                        vote_average = details['vote_average']
                        vote_count = details['vote_count']
                        director_popularity, cast_popularity = get_people_popularity(details['credits'])
                        genres = [genre['name'] for genre in details['genres']]
                        movie_popularity = details['popularity']
                        production_companies = [pc['name'] for pc in details['production_companies']]
                        budget = details['budget']
                        revenue = details.get('revenue', 0)
                        runtime = details['runtime']
                        tmdb_id = details['id']
                        imdb_id = details['imdb_id']
                        
                        row = [name, vote_average, vote_count, director_popularity, cast_popularity, genres, movie_popularity, production_companies, budget, revenue, runtime, release_date_month, tmdb_id, imdb_id]
                        no_votes_writer.writerow(row)
            page += 1  # Increment to the next page

# Run the function
fetch_upcoming_movies()
