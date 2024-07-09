
from flask import Flask, request, jsonify
from resume_filter import ResumeFilter

app = Flask(__name__)

# Initialize the ResumeFilter class
resume_filter = ResumeFilter()

@app.route('/filter', methods=['POST'])
def filter_resumes():
    data = request.json
    job_description = data.get('job_description')
    resumes = data.get('resumes')
    
    if not job_description or not resumes:
        return jsonify({'error': 'Job description and resumes are required'}), 400
    
    # Assume the ResumeFilter class has a method `filter_resumes`
    filtered_resumes = resume_filter.filter_resumes(job_description, resumes)
    
    return jsonify({'filtered_resumes': filtered_resumes})

@app.route('/status', methods=['GET'])
def status():
    return jsonify({'status': 'API is running'})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
