from vector_press.agent.agent import main

if __name__ == "__main__":
    main()




#TODO
# for now I need to convert database uploading method to a tool
# re-write a query generator like re-writing topic (especially topic because it has news, general, finance topics) , query, time_range etc.
# when you reached several tools track the llm's behaviour in debug mode to see tool_calls or other methods,
# bind tool u cikar onun yerine JSON schema'sini kendin yaz
# tool aciklamalarini vector database icerisine koy ve aciklamalari oradan cagir 10 numara context engineering, bir yontem daha var baska bir llm ile ya da ayni llm query sonrasi hangi tool(lar)a ihtiyacim var diye sorup kendince ihtiyac toolarini ortaya cikariyor
# context pruning yapilabilir, toplanan data icin (api den gelen sonuclar, eklenen dokumanlar vb.), zaten model bunu her turlu okuyup cost ediyor bunu okutturup ozet cikartirsak azicik daha harcar ama cok daha iyi sonuc cikarir.
# context offloading, texti dis bir notebook tarzi bir yerde tutmak. tam anlayamadim zamanla anlariz
# add newsapi for fetching news from another api source and run them parallel with guardian api to fetch news different sources (BBC API, New York Times API)
