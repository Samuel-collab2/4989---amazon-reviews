from textblob import TextBlob

def sentiment_analysis_tb(phrase):
    tb_phrase = TextBlob(phrase)

    print(tb_phrase.sentiment)
    # subjectivity score of 0 is objective, while 1 is very subjective

if __name__ == "__main__":
    # phrase = "Not one thing in this book seemed an obvious original thought. However, the clarity with which this author explains how innovation happens is remarkable. Alan Gregerman discusses the meaning of human interactions and the kinds of situations that tend to inspire original and/or clear thinking that leads to innovation. These things include how people communicate in certain situations such as when they are outside of their normal patterns. Gregerman identifies the ingredients that make innovation more likely. This includes people being compelled to interact when they normally wouldn't, leading to serendipity. Sometimes the phenomenon will occur through collaboration, and sometimes by chance such as when an individual is away from home on travel. I recommend this book for its common sense, its truth and the apparent mastery of the subject by the author."
    phrase = "This is a book of the ages and it is amazing."
    sentiment_analysis_tb(phrase)

    phrase = "This is book of ages and amazing."
    sentiment_analysis_tb(phrase)