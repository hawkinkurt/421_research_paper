import kagglehub

# Download latest version
path = kagglehub.dataset_download("raedaddala/imdb-movies-from-1960-to-2023")

print("Path to dataset files:", path)