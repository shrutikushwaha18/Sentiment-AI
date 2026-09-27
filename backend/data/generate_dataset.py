"""
generate_dataset.py
--------------------
Builds data/dataset.csv — a balanced, template + hand-curated sentiment
dataset with three classes: negative (0), neutral (1), positive (2).

This is a synthetic-but-realistic dataset meant for a working prototype
demo. Swap this file out for a real labeled dataset (e.g. IMDB, SST, Twitter
sentiment) for production-quality accuracy.
"""

import csv
import random

random.seed(42)

THINGS = [
    "movie", "film", "product", "service", "restaurant", "hotel", "app",
    "phone", "laptop", "book", "game", "trip", "flight", "concert", "show",
    "team", "meeting", "class", "course", "software", "update", "delivery",
    "customer support", "website", "meal", "coffee", "camera", "car",
    "experience", "presentation", "album", "song", "performance", "gym",
    "neighborhood", "apartment", "weather today", "traffic", "internet",
    "battery life", "user interface", "design", "story", "plot", "ending",
]

POS_ADJ = [
    "amazing", "fantastic", "wonderful", "excellent", "brilliant",
    "outstanding", "incredible", "superb", "delightful", "impressive",
    "fabulous", "great", "awesome", "terrific", "top-notch", "flawless",
    "refreshing", "charming", "remarkable", "phenomenal",
]

NEG_ADJ = [
    "terrible", "awful", "horrible", "disappointing", "frustrating",
    "mediocre", "dreadful", "pathetic", "useless", "annoying", "poor",
    "lousy", "atrocious", "unbearable", "subpar", "broken", "sluggish",
    "overpriced", "unreliable", "confusing",
]

NEU_ADJ = [
    "okay", "average", "fine", "decent", "standard", "ordinary",
    "acceptable", "typical", "unremarkable", "fair", "so-so",
]

POS_TEMPLATES = [
    "the {thing} was {adj}, I loved every moment of it",
    "I'm really happy with this {thing}, it exceeded my expectations",
    "what a {adj} {thing}, I would definitely recommend it to everyone",
    "this {thing} is {adj} and made my day so much better",
    "absolutely {adj} {thing}, best experience I've had in a while",
    "I can't stop smiling after that {adj} {thing}",
    "the {thing} worked perfectly and the team was incredibly helpful",
    "such a {adj} {thing}, everything went smoothly from start to finish",
    "I'm impressed, the {thing} is {adj} and worth every penny",
    "thank you so much, the {thing} made me so happy today",
    "the {thing} was {adj}, quick, and exactly what I needed",
    "wow this {thing} is {adj}, exceeded all my expectations",
    "great job on the {thing}, it was {adj} from start to end",
    "the {adj} {thing} left me feeling grateful and satisfied",
]

NEG_TEMPLATES = [
    "the {thing} was {adj}, I regret trying it",
    "I'm really disappointed with this {thing}, it did not meet expectations",
    "what a {adj} {thing}, I would not recommend it to anyone",
    "this {thing} is {adj} and ruined my entire day",
    "absolutely {adj} {thing}, worst experience I've had in a while",
    "I can't believe how {adj} that {thing} was",
    "the {thing} stopped working and support was completely unhelpful",
    "such a {adj} {thing}, everything went wrong from start to finish",
    "I'm furious, the {thing} is {adj} and a total waste of money",
    "I'm so upset, the {thing} ruined my whole day",
    "the {thing} was {adj}, slow, and nothing like what I expected",
    "ugh this {thing} is {adj}, fell short of every expectation",
    "terrible job on the {thing}, it was {adj} from start to end",
    "the {adj} {thing} left me feeling frustrated and let down",
]

NEU_TEMPLATES = [
    "the {thing} was {adj}, nothing special but nothing wrong either",
    "it's an {adj} {thing}, does the job without any surprises",
    "the {thing} was {adj}, I have mixed feelings about it",
    "this {thing} is {adj}, some parts were good and some were not",
    "the {thing} arrived on time, it was {adj} overall",
    "I have no strong opinion, the {thing} was just {adj}",
    "the {thing} is {adj}, might try it again, might not",
    "it was an {adj} {thing}, similar to what I've seen before",
    "the {thing} met the basic requirements, nothing more nothing less",
    "the {thing} was {adj}, I'm still deciding how I feel about it",
    "an {adj} {thing} overall, neither impressed nor disappointed",
    "the {thing} functioned as described, it was {adj}",
]

CURATED = [
    # positive
    ("I absolutely loved this movie, it was a masterpiece from start to finish.", 2),
    ("Best purchase I've made all year, works flawlessly!", 2),
    ("Customer service went above and beyond to help me, so grateful.", 2),
    ("This is hands down the best coffee shop in town.", 2),
    ("The concert last night was pure magic, unforgettable night.", 2),
    ("I'm thrilled with how the project turned out, great teamwork everyone!", 2),
    ("Five stars, would buy again without hesitation.", 2),
    ("The new update fixed everything and the app runs so smoothly now.", 2),
    ("What a beautiful sunny day, perfect for a walk in the park.", 2),
    ("Our vacation exceeded every expectation, we can't wait to go back.", 2),
    ("The chef outdid himself tonight, every dish was incredible.", 2),
    ("I finally passed my exam, so relieved and proud of myself!", 2),
    ("This book kept me hooked until the very last page.", 2),
    ("The staff were friendly, attentive, and made us feel so welcome.", 2),
    ("Great value for the price, exceeded my expectations completely.", 2),
    ("I'm so proud of the team, we crushed our quarterly goals.", 2),
    ("The new phone camera takes stunning photos, love it.", 2),
    ("Such a heartwarming film, I laughed and cried the whole way through.", 2),
    ("The delivery arrived early and everything was perfectly packaged.", 2),
    ("Highly recommend this place, the ambiance and food were fantastic.", 2),
    # negative
    ("This is the worst customer service I have ever experienced.", 0),
    ("The product broke after just two days, total waste of money.", 0),
    ("I regret watching this movie, it was boring and way too long.", 0),
    ("The food was cold and tasteless, we will never go back.", 0),
    ("My flight got delayed for six hours with zero communication from the airline.", 0),
    ("The app keeps crashing and I've lost all my saved data.", 0),
    ("Extremely disappointed with the quality, it looks nothing like the pictures.", 0),
    ("The hotel room was dirty and smelled awful, ruined our trip.", 0),
    ("I was overcharged and support refuses to give me a refund.", 0),
    ("This laptop overheats constantly and the battery barely lasts an hour.", 0),
    ("Rude staff, slow service, and the order was completely wrong.", 0),
    ("The concert was a disaster, terrible sound and the artist left early.", 0),
    ("I can't believe how unreliable this software has become lately.", 0),
    ("The meeting was a complete waste of time, nothing got decided.", 0),
    ("Package arrived damaged and customer support has ignored my emails.", 0),
    ("The plot made no sense and the acting was painfully bad.", 0),
    ("Traffic was horrendous today, sat in my car for two hours.", 0),
    ("I'm so frustrated, the website crashes every time I try to checkout.", 0),
    ("The course was disorganized and the instructor never answered questions.", 0),
    ("Worst meal I've had in years, definitely not coming back here.", 0),
    # neutral
    ("The package arrived on the estimated delivery date.", 1),
    ("The meeting is scheduled for 3pm tomorrow in the main conference room.", 1),
    ("The movie was okay, not great but not terrible either.", 1),
    ("It's a decent laptop for basic tasks like browsing and email.", 1),
    ("The restaurant serves standard Italian food, similar to most places.", 1),
    ("The report covers sales figures for the last three quarters.", 1),
    ("I'm not sure how I feel about the new update yet.", 1),
    ("The weather today is partly cloudy with a chance of rain.", 1),
    ("The hotel room was average, clean but nothing memorable.", 1),
    ("The book has an interesting premise but a slow middle section.", 1),
    ("The team met the deadline, results were within expected range.", 1),
    ("This phone has the features I need, nothing extra though.", 1),
    ("The service was fine, no complaints but nothing stood out either.", 1),
    ("The store is open from nine in the morning until six in the evening.", 1),
    ("It's an average camera, good enough for casual photos.", 1),
    ("The flight departed and landed on schedule.", 1),
    ("The course covers the basics of the subject in six weeks.", 1),
    ("The app works as described, does what it says.", 1),
    ("The presentation covered the quarterly numbers as planned.", 1),
    ("It's a typical Monday, nothing unusual happened at work.", 1),
]


def generate_templated(n_per_class: int):
    rows = []
    for templates, adjs, label in [
        (POS_TEMPLATES, POS_ADJ, 2),
        (NEG_TEMPLATES, NEG_ADJ, 0),
        (NEU_TEMPLATES, NEU_ADJ, 1),
    ]:
        seen = set()
        count = 0
        attempts = 0
        while count < n_per_class and attempts < n_per_class * 20:
            attempts += 1
            t = random.choice(templates)
            thing = random.choice(THINGS)
            adj = random.choice(adjs)
            sentence = t.format(thing=thing, adj=adj)
            if sentence in seen:
                continue
            seen.add(sentence)
            rows.append((sentence, label))
            count += 1
    return rows


def main():
    rows = generate_templated(n_per_class=220)
    rows.extend(CURATED)
    random.shuffle(rows)

    out_path = "dataset.csv"
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["text", "label"])
        writer.writerows(rows)

    counts = {0: 0, 1: 0, 2: 0}
    for _, label in rows:
        counts[label] += 1
    print(f"Wrote {len(rows)} rows to {out_path}")
    print(f"Negative: {counts[0]}, Neutral: {counts[1]}, Positive: {counts[2]}")


if __name__ == "__main__":
    main()
