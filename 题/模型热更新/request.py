import requests

r = requests.post('http://localhost:5000/predict', json={'input': [[.1, .2, .3, .4, .5], [.5, .4, .3, .2, .1]]})
print(r.json())
# {'current_model': 'svm_1.pkl', 'output': [0, 0]}

r = requests.post('http://localhost:5000/update', json={'model': 'svm_2.pkl'})
print(r.status_code)
# 200

r = requests.post('http://localhost:5000/predict', json={'input': [[.1, .2, .3, .4, .5], [.5, .4, .3, .2, .1]]})
print(r.json())
# {'current_model': 'svm_2.pkl', 'output': [1, 1]}

r = requests.post('http://localhost:5000/update', json={'model': 'svm_4.pkl'})
print(r.status_code)
# 400