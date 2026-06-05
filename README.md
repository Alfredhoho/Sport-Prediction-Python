# Sport Prediction
**A Student Project for Predicting Football Games Based off Game Data and Analysis via an AI Agent.**

## How it works
Through [Football-Data.co.uk](Football-Data.co.uk) we're able to get CSV files containing historical and current match results.   Using that data, with Maher Poisson inspired prediction model[^1] we are able to to predict the results of future games that the end user specifies.  
For the AI powered feature, through ChatGPT's API we're get an explaination on why this result is likely.
[^1]: Changes made to this model to include exponential weighting so recent matches matter more.


## How to Run it Locally

cd backend  
python -m venv .venv  
source .venv/bin/activate  
pip install -r requirements.txt  
cp .env.example .env  
python scripts/download_football_data.py  
uvicorn app.main:app --reload  

Open frontend/index.html and use http://127.0.0.1:8000 as the backend URL.

## Video Demo



https://github.com/user-attachments/assets/d85c8ed8-30f2-4bcb-a4ee-7054420b32a3




## Research Paper
[Maher, M. J. (1982). "Modelling association football scores."](http://www.90minut.pl/misc/maher.pdf) The paper models football scores using Poisson distributions and team attack/defense strengths. We

## AI Use and Agent Reflection
*We did not use the AI as the main predictor. The prediction comes from the statistical model trained on former soccer results. The AI API is used only to explain the prediction in a more understandable way.*

What you used the agent for is specific. Which features did you build with its help? What kinds of prompts did you give it?  
> For this project we asked the agent to create the code for the frontend, specifically the HTML and Script aspect. The prompt we used was simple, "Generate a frontend that for the match predictor". After it gave us the basic model we edited the build to our liking and tweaked the `style.css` for the website. For the backend the agent was responsible for implementing the AI features and helping us refine the ability for the frontend to call the backend and fixing mistakes in `model.py` where we used the Poisson model.
  
What worked well where did the agent save you time or produce something better than you expected?  
> Having little experience in HTML and frontend, the agent created a good looking website, and not only saved us time but also gave us an explaination on what it did, how it did it, and how does it work.

What didn't work, where did it make mistakes, go in the wrong direction, or require significant correction?  
> The agent made assumptions and made code that we didn't specify and we had to overhaul code that we previously though was working.

What you learned about prompting, what did you figure out about how to ask it effectively?  
> Specific prompts get much better results. A broad prompt like “make a soccer predictor” produces a generic app or an AI wrapper. It's also not good to rely on Ai for everything, especially for the design aspect. The AI works best when we make the design decisions and use the agent to help implement them. We still needed to understand the code and test the front and backend.

What you would do differently knowing what you know now, how would you change the way you collaborated with it?  
> Knowing what we do now, we would've created a better design and plan instead of going into the project essentially blind and not having a clear vision on what we wanted the AI to accomplish in terms of both helping write the scripts and in the AI feature.
