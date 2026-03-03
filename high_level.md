Week 1 Meeting - 01 Jan 26


Where we are now

Mostly automatic pipeline that takes in a dataset and:
Uses GPT-5-mini to give info on all metadata, provide recommendation on preprocessing steps and model training
Followed by execution of recommended preprocessing steps
Lastly runs a trainer that applies ML models if our codebase has already defined them


This Semester:
Improve pipeline
Define more ML models (somehow get pre-trained if we can)
Tune and make ML models for better performance
Implement different AI agents for different steps (find which are best at what)
Use AI recommendation on non-deterministic steps
Create Agents/Engines for other tasks
Ingestion Agent (LLM metadata extraction, create a dataset profile)
Planning Agent (take dataset profile and recommend preprocessing, transformers, ML models, tuning (optuna), split, etc)
Training/Testing Agent
Eval Agent (metrics and comparison for dashboard)

Ideas:
Create catalogs for agents to ‘browse’
i.e transformers, models, splits catalogs. Agents will refer to this and recommend from these catalogs


Goal of semester:
Have a live web based UI. Upload dataset and get a dashboard of results and comparisons between models/methods applied and explain which is the best approach for given dataset.




Possible Assigned Work:
Jared - Ingestion Agent
Wes - Build/Define models / training testing agent
Saoud - Planning Agent
Tommy - Build/Define models / training testing agent
Sam - Eval Agent

Meeting missing Saoud and Sam due to conflicts. Roles not confirmed and agents to be more defined and planned.







