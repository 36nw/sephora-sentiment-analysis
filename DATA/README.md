## Dataset Source:
https://www.kaggle.com/datasets/nadyinky/sephora-products-and-skincare-reviews

## Dataset Overwiev

The dataset is the “Sephora Products and Skincare Reviews (2023)” obtained from the Kaggle website. 
This dataset consists of product reviews from Sephora online store and over 1 million reviews on over 2,000 products from the Skincare category, including user appearances, and review ratings by other users (Inky 2023), 
providing a rich source of textual data for sentiment analysis.
  
For sentiment analysis, the relevant features/columns for analysis include the “review_text”, “rating”, “reviews”, and “product_name”. 
These features can provide contextual information and contribute to the sentiment analysis process and text visualization.
  
The dataset consists of product information and user reviews in CSV format. The product information contains details about over 8,000 beauty products from the Sephora online store.
The user review CSV files contain over 1 million reviews specifically for skincare products. These reviews are spread across the six CSV files and provide insights into user experiences with the products.

## Data Attributes Descriptions

The product information consists of the following 12 features:
  
- product_id: A unique identifier for each product from the site.
- product_name: The full name of the product.
- brand_id: A unique identifier for the product's brand from the site.
- brand_name: The full name of the product's brand.
- loves_count: The number of people who have marked the product as a favorite.
- rating: The average rating of the product based on user reviews.
- reviews: The number of user reviews for the product.
- size: The size of the product, which may be in ounces (oz), milliliters (ml), grams (g), packs, or other relevant units depending on the product type.
- variation_type: The type of variation parameter for the product, such as size or color.
- variation_value: The specific value of the variation parameter for the product, for example, "100 mL" or "Golden Sand."
- variation_desc: A description of the variation parameter for the product, such as the tone for the fairest skin.
- ingredients: A list of ingredients included in the product. If there are variations, the list may include the specific ingredients for each variation.
  
The reviews consist of the following 15 features:
  
- author_id: A unique identifier for the author of the review on the website.
- rating: The rating given by the author for the product on a scale of 1 to 5.
- is_recommended: Indicates if the author recommends the product or not (1 for true, 0 for false).
- helpfulness: The ratio of positive ratings to the total number of ratings for the review. It is calculated as total_pos_feedback_count / total_feedback_count.
- total_feedback_count: The total number of feedback (positive and negative ratings) left by users for the review.
- total_neg_feedback_count: The number of users who gave a negative rating for the review.
- total_pos_feedback_count: The number of users who gave a positive rating for the review.
- submission_time: The date the review was posted on the website in the 'yyyy-mm-dd' format.
- review_text: The main text of the review written by the author.
- review_title: The title of the review written by the author.
- skin_tone: The author's skin tone, such as fair, tan, etc.
- eye_color: The author's eye color, such as brown, green, etc.
- skin_type: The author's skin type, such as combination, oily, etc.
- hair_color: The author's hair color, such as brown, auburn, etc.
- product_id: The unique identifier for the product on the website.
