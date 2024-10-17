from flask import Flask, render_template, redirect, url_for, request, flash, send_file, jsonify
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user
from flask_mysqldb import MySQL
from werkzeug.security import generate_password_hash, check_password_hash
from datetime import datetime
import csv
import os
import subprocess
from threading import Thread

app = Flask(__name__)
app.secret_key = 'your_secret_key'

# Database configuration
app.config['MYSQL_HOST'] = 'localhost'
app.config['MYSQL_USER'] = 'hokengakari'
app.config['MYSQL_PASSWORD'] = '230073'
app.config['MYSQL_DB'] = 'shoubou_data'
mysql = MySQL(app)

login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

class User(UserMixin):
    def __init__(self, username):
        self.id = username

def load_users():
    users = {}
    try:
        with open('static/users.csv', mode='r', encoding='utf-8') as file:
            reader = csv.DictReader(file)
            for row in reader:
                users[row['username']] = row['password']
    except Exception as e:
        print(f"Error reading users file: {e}")
    return users

@login_manager.user_loader
def load_user(user_id):
    users = load_users()
    return User(user_id) if user_id in users else None

@app.route('/')
@login_required
def home():
    return render_template('index.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        
        users = load_users()
        if username in users:
            if check_password_hash(users[username], password):
                login_user(User(username))
                return redirect(url_for('home'))
            else:
                flash('Invalid credentials.')
        else:
            flash('Invalid credentials.')
    return render_template('login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        
        users = load_users()
        if username in users:
            flash('Username already exists.')
        else:
            hashed_password = generate_password_hash(password)
            try:
                with open('static/users.csv', mode='a', encoding='utf-8', newline='') as file:
                    writer = csv.DictWriter(file, fieldnames=['username', 'password'])
                    writer.writerow({'username': username, 'password': hashed_password})
                flash('Registration successful! You can now log in.')
                return redirect(url_for('login'))
            except Exception as e:
                print(f"Error writing to users file: {e}")

    return render_template('register.html')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('home'))

@app.route('/add', methods=['GET', 'POST'])
@login_required
def add():
    if request.method == 'POST':
        cursor = mysql.connection.cursor()
        try:
            year_month = request.form['weather']
            date_obj = datetime.strptime(year_month, '%Y-%m-%d')
            week_day = date_obj.strftime('%A')

            weekday_map = {
                'Monday': '月',
                'Tuesday': '火',
                'Wednesday': '水',
                'Thursday': '木',
                'Friday': '金',
                'Saturday': '土',
                'Sunday': '日'
            }

            week_day_japanese = weekday_map[week_day]

            weather = request.form['weather']
            place = request.form['place']
            gender = request.form['gender']
            age_group = request.form['age_group']

            cursor.execute('''INSERT INTO all_datas (覚知年月日, 覚知曜日, 天候, 出場場所地区, 性別, 年齢区分_サーベイランス用) 
                              VALUES (%s, %s, %s, %s, %s, %s)''', 
                           (year_month, week_day_japanese, weather, place, gender, age_group))
            mysql.connection.commit()
            flash('データが正常に送信されました！')
        except KeyError as e:
            mysql.connection.rollback()
            flash('Required form field is missing.')
        except Exception as e:
            mysql.connection.rollback()
            flash('データの送信に失敗しました。')
        finally:
            cursor.close()

        return redirect(url_for('add'))

    return render_template('add.html')

@app.route('/map')
@login_required
def map_page():
    return render_template('map.html')
@app.route('/run-gg', methods=['POST'])
def run_gg_script():
    data = request.get_json()
    input_date = data.get('date')  # Get the date from the request body
    
    # Call gg.py with the input date
    try:
        result = subprocess.run(['python', 'gg.py', input_date], check=True, capture_output=True, text=True)
        return jsonify({"message": "Script executed successfully.", "output": result.stdout}), 200
    except subprocess.CalledProcessError as e:
        return jsonify({"error": str(e), "output": e.output}), 500

@app.route('/run', methods=['POST'])
def run_script():
    def run_mouse_script():
        try:
            subprocess.run(['python', 'mouse_click.py'])
        except Exception as e:
            print(f"Error executing mouse_click.py: {e}")

    Thread(target=run_mouse_script).start()
    return jsonify(message="Script is running"), 200

@app.route('/calendar')
@login_required
def calendar():
    return render_template('calendar.html')

@app.route('/data')
@login_required
def data():
    trends_file_path = os.path.join('static', 'trends.csv')

    if not os.path.isfile(trends_file_path):
        flash('The trends.csv file is not found.')
        return redirect(url_for('home'))

    return send_file(trends_file_path, mimetype='text/csv', as_attachment=True)

@app.route('/contact')
def contact():
    return render_template('contact.html')

if __name__ == '__main__':
    app.run(debug=True)
