from vector_press.agent.agent import main

if __name__ == "__main__":
    main()




#TODO
# for now I need to convert database uploading method to a tool
# re-write a query generator like re-writing topic (especially topic because it has news, general, finance topics) , query, time_range etc.
# update the streamlit_interface and check if langsmith is working
# when you reached several tools track the llm's behaviour in debug mode to see tool_calls or other methods,
# bind tool u cikar onun yerine JSON schema'sini kendin yaz