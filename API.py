from array import array
from googleapiclient.discovery import build 
from google_auth_oauthlib.flow import InstalledAppFlow 
from google.auth.transport.requests import Request 
import pickle 
import os.path 
import base64 
import re
from bs4 import BeautifulSoup 

SCOPES = ['https://www.googleapis.com/auth/gmail.readonly'] 

urls = []

def extract_urls_from_text(text):
    urls = re.findall(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', text)
    return urls

def getEmails(): 
	creds = None

	if os.path.exists('token.pickle'): 

		with open('token.pickle', 'rb') as token: 
			creds = pickle.load(token) 

	if not creds or not creds.valid: 
		if creds and creds.expired and creds.refresh_token: 
			creds.refresh(Request()) 
		else: 
			flow = InstalledAppFlow.from_client_secrets_file('credentials.json', SCOPES) 
			creds = flow.run_local_server(port=8080) 

		with open('token.pickle', 'wb') as token: 
			pickle.dump(creds, token) 

	service = build('gmail', 'v1', credentials=creds) 

	result = service.users().messages().list(userId='me').execute() 

	result = service.users().messages().list(maxResults=2, userId='me').execute() 
	messages = result.get('messages') 

	for msg in messages: 
		try:
			txt = service.users().messages().get(userId='me', id=msg['id']).execute() 

			payload = txt['payload'] 
			headers = payload['headers'] 

			# for d in headers: 
			# 	if d['name'] == 'Subject': 
			# 		subject = d['value'] 
			# 	if d['name'] == 'From': 
			# 		sender = d['value'] 

			parts = payload.get('parts')[0] 
			data = parts['body']['data'] 
			data = data.replace("-","+").replace("_","/") 
			decoded_data = base64.b64decode(data) 

			soup = BeautifulSoup(decoded_data , "lxml") 
			body = soup.body() 

			urls = extract_urls_from_text(str(body))
			# print("Message: ", body)
			# print("Extracted URLs: ", urls)
			# print('\n')
			return urls
		except:
			pass
getEmails()