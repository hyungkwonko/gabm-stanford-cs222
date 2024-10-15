from simulation_engine.settings import *
from simulation_engine.global_methods import *

from agent_bank.navigator import *
from generative_agent.generative_agent import *

from cs222_assignment_1.memories.jasmine_carter_memories import *
from cs222_assignment_1.memories.matthew_jacobs_memories import *
from cs222_assignment_1.questions.jasmine_carter_questions import *
from cs222_assignment_1.questions.matthew_jacobs_questions import *

import pandas as pd

POPULATION_BASE = "SyntheticCS222_Base"
POPULATION = "SyntheticCS222"
AGENT_ID = "matthew_jacobs"
AGENT_FOLDER = f"{POPULATIONS_DIR}/{POPULATION_BASE}/{AGENT_ID}"


q_dir = f"{BASE_DIR}/cs222_assignment_1/report/{AGENT_ID}/questions__{AGENT_ID}.csv"
a_dir = f"{BASE_DIR}/cs222_assignment_1/report/{AGENT_ID}/answers__{AGENT_ID}.csv"


def chat_session(generative_agent):
    questions_df = pd.read_csv(q_dir, header=None)
    questions = questions_df.iloc[:, 0].tolist()

    qa_pairs = []

    for i, question in enumerate(questions):
        curr_convo = []
        curr_convo += [[KEY_OWNER, question]]

        response = generative_agent.utterance(curr_convo)
        curr_convo += [[generative_agent.scratch.get_fullname(), response]]

        print(f"({i}): {KEY_OWNER}: {question}")
        print(f"({i}): {generative_agent.scratch.get_fullname()}: {response}")
        print("-" * 10)

        qa_pairs.append([question, response])

    qa_df = pd.DataFrame(qa_pairs)
    qa_df.to_csv(a_dir, index=False, header=False)


def build_agent():
    curr_agent = GenerativeAgent(POPULATION_BASE, AGENT_ID)
    for m in matthew_memories:
        curr_agent.remember(m)
    curr_agent.save(POPULATION, AGENT_ID)


def interview_agent():
    curr_agent = GenerativeAgent(POPULATION, AGENT_ID)
    chat_session(curr_agent)


def main():

    # We stop the process if the agent storage folder already exists.
    if not check_if_file_exists(f"{AGENT_FOLDER}/scratch.json"):
        print("Make new generative agent for given location!")
        build_agent()
    interview_agent()


if __name__ == "__main__":
    main()
