from flask import Flask, render_template, request, redirect, url_for, send_from_directory, session
import pandas as pd, os, math, numpy as np
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import pairwise_distances

app = Flask(__name__)
app.secret_key = 'replace_this_with_a_secure_random_value'
UPLOAD_CSV = 'assigned_rooms.csv'

# --- Helper functions ---
def normalize_col_name(s):
    return s.strip().lower().replace('_',' ').replace('-',' ').replace('.', ' ')

def find_columns(df):
    cols = {normalize_col_name(c): c for c in df.columns}
    name_col = cols.get('name')
    roll_col = cols.get('roll no') or cols.get('roll') or cols.get('roll number') or cols.get('rollno')
    feature_cols = []
    for key in ['sleep time','study time','noise tolerance']:
        if key in cols:
            feature_cols.append(cols[key])
    password_col = cols.get('password') or cols.get('pass') or cols.get('passwd')
    return name_col, roll_col, feature_cols, password_col

# --- Routes ---
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/about')
def about():
    return render_template('about.html')  # <-- About Us page route

@app.route('/admin_login', methods=['GET','POST'])
def admin_login():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        if username == 'admin' and password == 'admin123':
            session['admin_logged_in'] = True
            return redirect(url_for('admin_panel'))
        return render_template('admin_login.html', error='Invalid credentials')
    return render_template('admin_login.html')

@app.route('/admin_panel')
def admin_panel():
    if not session.get('admin_logged_in'):
        return redirect(url_for('admin_login'))
    return render_template('admin_panel.html')

@app.route('/logout')
def logout():
    session.pop('admin_logged_in', None)
    session.pop('student_authenticated', None)
    session.pop('student_roll', None)
    return redirect(url_for('index'))

# --- CSV Upload & Room Assignment ---
@app.route('/upload_csv', methods=['POST'])
def upload_csv():
    if not session.get('admin_logged_in'):
        return redirect(url_for('admin_login'))
    f = request.files.get('file')
    if not f:
        return "No file uploaded. <a href='/admin_panel'>Go back</a>"
    try:
        df = pd.read_csv(f)
    except Exception as e:
        return f'Failed to read CSV: {e}. <a href="/admin_panel">Go back</a>'
    name_col, roll_col, feature_cols, password_col = find_columns(df)
    if not name_col or not roll_col or len(feature_cols) < 1:
        return ("CSV missing required columns. Make sure CSV has: 'Name, roll no, sleep time, study time, noise tolerance' <a href='/admin_panel'>Go back</a>")
    df = df.reset_index(drop=True).copy()
    if not password_col:
        df['password'] = df[roll_col].astype(str)
        password_col = 'password'
    else:
        if password_col != 'password':
            df = df.rename(columns={password_col: 'password'})
            password_col = 'password'
    Xcols = []
    for c in feature_cols:
        series = df[c]
        snum = pd.to_numeric(series, errors='coerce')
        if snum.isna().any():
            uniq = list(sorted(series.dropna().unique(), key=str))
            mapping = {v:i for i,v in enumerate(uniq)}
            enc = series.map(lambda v: mapping.get(v, 0)).astype(float)
            df[c + '_enc'] = enc
            Xcols.append(c + '_enc')
        else:
            df[c] = snum.astype(float)
            Xcols.append(c)
    X = df[Xcols].values
    if X.shape[0] == 0:
        return 'No data rows found in CSV.'
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    n_students = Xs.shape[0]
    n_rooms_target = math.ceil(n_students / 2)
    clustering = AgglomerativeClustering(n_clusters=n_rooms_target, linkage='ward')
    labels = clustering.fit_predict(Xs)
    df['Cluster'] = labels
    idx_to_room = {}
    current_room = 101
    for cl in sorted(np.unique(labels)):
        idxs = list(np.where(labels == cl)[0])
        while len(idxs) >= 2:
            i = idxs.pop(0)
            dists = pairwise_distances(Xs[idxs], Xs[i].reshape(1,-1)).flatten()
            j_pos = int(np.argmin(dists))
            j = idxs.pop(j_pos)
            idx_to_room[i] = current_room
            idx_to_room[j] = current_room
            current_room += 1
        if len(idxs) == 1:
            i = idxs.pop(0)
            idx_to_room[i] = current_room
            current_room += 1
    for i in range(n_students):
        if i not in idx_to_room:
            idx_to_room[i] = current_room
            current_room += 1
    df['Room'] = df.index.map(lambda i: idx_to_room.get(i))
    df.to_csv(UPLOAD_CSV, index=False)
    return "CSV uploaded and rooms assigned successfully! <br><a href='/results'>View Results</a>"

@app.route('/results')
def results():
    if not session.get('admin_logged_in'):
        return redirect(url_for('admin_login'))
    if not os.path.exists(UPLOAD_CSV):
        return "No results yet. <a href='/admin_panel'>Go back</a>"
    df = pd.read_csv(UPLOAD_CSV)
    students = df.to_dict(orient='records')
    return render_template('results.html', students=students)

@app.route('/download_assigned')
def download_assigned():
    if not session.get('admin_logged_in'):
        return redirect(url_for('admin_login'))
    if os.path.exists(UPLOAD_CSV):
        return send_from_directory('.', UPLOAD_CSV, as_attachment=True)
    return "No CSV available."

# --- Student login & dashboard ---
@app.route('/student', methods=['GET','POST'])
def student_login():
    error = None
    if request.method == 'POST':
        roll = request.form.get('roll')
        password = request.form.get('password')
        if not os.path.exists(UPLOAD_CSV):
            error = 'No assignments yet. Ask admin to upload CSV.'
            return render_template('student_login.html', error=error)
        df = pd.read_csv(UPLOAD_CSV)
        mapping = {normalize_col_name(c): c for c in df.columns}
        roll_candidates = ['roll no','roll','rollno','roll number']
        roll_col = None
        for rc in roll_candidates:
            if rc in mapping:
                roll_col = mapping[rc]
                break
        if roll_col is None:
            error = 'Roll column not found in assigned CSV'
            return render_template('student_login.html', error=error)
        matched = df[df[roll_col].astype(str) == str(roll)]
        if matched.empty:
            error = 'Roll number not found'
            return render_template('student_login.html', error=error)
        student = matched.iloc[0]
        pwd_col = 'password' if 'password' in df.columns else None
        stored_pwd = str(student[pwd_col]) if pwd_col else str(student[roll_col])
        if str(password) != stored_pwd:
            error = 'Incorrect password'
            return render_template('student_login.html', error=error)
        session['student_roll'] = str(roll)
        session['student_authenticated'] = True
        return redirect(url_for('student_dashboard', roll=roll))
    return render_template('student_login.html', error=error)

@app.route('/student_dashboard/<roll>')
def student_dashboard(roll):
    if not session.get('student_authenticated') or session.get('student_roll') != str(roll):
        return redirect(url_for('student'))
    if not os.path.exists(UPLOAD_CSV):
        return 'No assignments yet. Admin must upload CSV. <a href="/admin_login">Go Back</a>'
    df = pd.read_csv(UPLOAD_CSV)
    mapping = {normalize_col_name(c): c for c in df.columns}
    name_col = mapping.get('name')
    roll_candidates = ['roll no','roll','rollno','roll number']
    roll_col = None
    for rc in roll_candidates:
        if rc in mapping:
            roll_col = mapping[rc]
            break
    matched = df[df[roll_col].astype(str) == str(roll)]
    if matched.empty:
        return 'Student not found. Check roll number. <a href="/student">Go Back</a>'
    student = matched.iloc[0]
    roommates = df[df['Room'] == student['Room']][name_col].tolist()
    roommates = [r for r in roommates if r != student[name_col]]
    return render_template('student_dashboard.html', student_name=student[name_col], room_number=student['Room'], roommates=roommates)

# --- Run app ---
if __name__ == '__main__':
    app.run(debug=True)
