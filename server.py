from flask import Flask, render_template, request
import pickle
from sentiment_model_ML import clean_reviews
import sqlite3
app = Flask('__name__')

conn = sqlite3.connect('database.db')
print ("Opened database successfully")

conn.execute('CREATE TABLE if not exists sentiment_response_db (id INTEGER PRIMARY KEY AUTOINCREMENT,movie_review TEXT, sentiment TEXT, feedback TEXT)')
print ("Table created successfully")
conn.close()

model = pickle.load(open('sentiment_model_ML/model.pkl', 'rb'))
vectorizer = pickle.load(open('sentiment_model_ML/vectorizer.pkl', 'rb'))

@app.route('/')
@app.route('/home')
def home_page():
    return render_template('home.html')

@app.route('/result', methods=['POST'])
def result_page():
    req_data = request.form['review_text']
    print('Req data: ',req_data)
    cleaned_rev = clean_reviews.review_cleaner(req_data)
    test_bag = vectorizer.transform([cleaned_rev]).toarray()
    prediction = model.predict(test_bag)
    print('Prediction val: ',prediction)
    if(prediction == 0):
        sentiment = 'Negative'
    elif(prediction == 1):
        sentiment = 'Positive'
    try:
        conn = sqlite3.connect('database.db')
        cur = conn.cursor()
        cur.execute("INSERT INTO sentiment_response_db (movie_review,sentiment) VALUES (?,?)",(req_data,sentiment))
        conn.commit()
        review_id = cur.lastrowid
        return render_template('result.html', review=req_data, sentiment=sentiment, review_id=review_id)
    except:
        print('Some error occurred')
        conn.rollback()
    finally:
        conn.close()

@app.route('/feedback', methods=['POST'])
def feedback_page():
    feedback_val = request.form['feedback_val']
    review_id = request.form['review_id']
    print('Review id: ',review_id, 'Feedback: ',feedback_val)
    try:
        conn = sqlite3.connect('database.db')
        cur = conn.cursor()
        cur.execute("update sentiment_response_db set feedback=? where id=?", (feedback_val, review_id))        
        conn.commit()
        conn.row_factory = sqlite3.Row
        #cur = conn.cursor()
        cur.execute("select * from sentiment_response_db where id=?",(review_id))
        rows = cur.fetchall()  
        print('Rowwws: ',rows)      
        return render_template('feedback.html', rows=rows)
    except:
        print('Some error occurred')
        conn.rollback()
    finally:
        conn.close()
    
@app.route('/data')
def data_page():
    conn = sqlite3.connect("database.db")
    conn.row_factory = sqlite3.Row

    cur = conn.cursor()
    cur.execute("select * from sentiment_response_db")

    rows = cur.fetchall() # returns list of dictionaries
    return render_template("data.html",rows = rows)

if __name__ == '__main__':
    # Threaded option to enable multiple instances for multiple user access support
    app.run()