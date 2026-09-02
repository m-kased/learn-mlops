import pandas as pd

s3_url = "s3://mobbucketsagemaker-jx-2025/data/newsCorpora.csv"

# UCI News Aggregator: tab-separated, no header. Commas in titles/URLs
# make the default csv parser fail (e.g. "Expected 4 fields, saw 5").
    df = pd.read_csv(
        s3_url,
        sep="\t",
        header=None,
        names=["id", "title", "url", "publisher", "category", "story", "hostname", "timestamp"],
    )
print(df.head())
print(df.shape)