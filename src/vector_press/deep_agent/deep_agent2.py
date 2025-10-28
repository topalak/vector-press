import os
from pyexpat import model

from config import settings

from src.vector_press.model_config import ModelConfig
from src.config import settings
from src.vector_press.agent.tools import Tools

from deepagents import create_deep_agent

llm_config = ModelConfig(model="gpt-oss:120b-cloud", use_cloud=True, api_key=settings.OLLAMA_API_KEY, num_ctx=8192)
                         #api_key=settings.OLLAMA_API_KEY)
llm = llm_config.get_llm()

embedding_config = 3 #we need to add here a runnable config

#ilk stateini ve config ini freeze ediyor, sonra ikincinin stateine geciyor asinda bunlar ayni class icinde yer aliyolar ama farkli icerikler tabii ki, ikinci node da ikien ilkinden sonuc
#geldi ilk freeze edilen state i tekrar cagiriyor, kullandigimz graph compile aslinda bu dondurulmayi aciyor aslinda onun gibi bir sey, normalde icine bir sey vermiyoruz ama bu durumda 1. node
#dan gelen statei pass ediyormusuz gibi dusunelim yani oradan devam ediyor. diyelim ki su an 3. node u cagirdi ve o sirada 1. node da yer alan llm cevabi geldi ve event loop sunu dedi: dur hemen
#dondurabilecegim yerde durdurup sana gelicem diyip sonra gelip 3. nun state ine kaldigi yerden devam ediyor.

#persistant state

#structured tool aslinda structured output, __init__, llm cagirriktan sonra cagrilacak tool olarak tanimliyor asinda senin pydantic classinin init methodunu llm.with_structured_output bind tools methodu cagriliypr


# Instantiate Tools
tools_instance = Tools()


################# AGENT 2 ####################

sub_critique_description = """ 
Used to critique the final report. This agent needs to invoke right after 'report_writer_agent' to check is the final report is excellent.
Don't forget to call 'critique_agent' after the 'report_writer_agent' is called.
"""

sub_critique_prompt = """You are a dedicated editor. You are being tasked to critique a report.

You can find the report at `final_report.md`.

You can find the question/topic for this report at `question.txt`.

You can use the search tool to search for information, if that will help you critique the report

Do not write to the `final_report.md` yourself.

Things to check:
- Check that each section is appropriately named
- Check that the report is written as you would find in an essay or a textbook - it should be text heavy, do not let it just be a list of bullet points!
- Check that the report is comprehensive. If any paragraphs or sections are short, or missing important details, point it out.
- Check that the article covers key areas of the industry, ensures overall understanding, and does not omit important parts.
- Check that the article deeply analyzes causes, impacts, and trends, providing valuable insights
- Check that the article closely follows the research topic and directly answers questions
- Check that the article has a clear structure, fluent language, and is easy to understand.
"""

sub_critique_agent = {
    "name": "critique_agent",
    "description": sub_critique_description,
    "system_prompt": sub_critique_prompt,
}



base_agent_instructions =  """You are a professional researcher, managing tool and deliver comprehensive, checked reports.
    Your Tools:

  **guardian_api**: Search The Guardian for reliable news
    When to use:
    - News about world events, politics, current affairs
    - Business news, economics, corporate stories
    - Culture, lifestyle, opinion pieces
    - ARCHIVED news articles (Guardian has extensive archives)
    - Any time you need reliable, journalistic sources

  **web_search_tool**: Search web with Tavily
    When to use:
    - Tutorials, guides, how-to information (e.g., "how to learn Python")
    - Historical information (e.g., "history of Bitcoin", "what is quantum computing")
    - Concepts, definitions, explanations (e.g., "explain blockchain")
    - Financial market data or analysis (set topic='finance')
    - Recent updates not yet in Guardian

  **sports_rss_feed**: Fetch current sports news topics, There will be a lot of title, just select most valuable 3 - 5 topics and their links.
  **technology_rss_feed**: Fetch current tech news topics, There will be a lot of title, just select most valuable 3 - 5 topics and their links.

  **critique_tool**: Critique the final result for quality assurance, you MUST use that tool before delivering the output.
  
    When to use:
    - MUST call it right after generating your final result
    - Use to check if the final report is excellent and comprehensive
    - Validates structure, clarity, completeness, and accuracy
    - Don't forget to call this tool after finishing your report


  You are going to write a final report by applying these steps:
    
    If base agent wants detailed report follow these:
        Please create a detailed answer to the overall research brief that:
        1. Is well-organized with proper headings (# for title, ## for sections, ### for subsections)
        2. Includes specific facts and insights from the research
        3. References relevant sources using [Title](URL) format, just share the most valuable 3 or 4 source, do not share every source. Its IMPORTANT. 
        4. Provides a balanced, thorough analysis. Be as comprehensive as possible, and include all information that is relevant to the overall research question. People are using you for deep research and will expect detailed, comprehensive answers.
        5. Includes a "Sources" section at the end.
        
  Otherwise keep your answer SIMPLE.  

  You can structure your report in a number of different ways. Here are some examples:
        
    To answer a question that asks you to return a list of things, you might only need a single section which is the entire list.
    1/ list of things or table of things
    Or, you could choose to make each item in the list a separate section in the report. When asked for lists, you don't need an introduction or conclusion.
    /intro
    1/ item 1
    2/ item 2
    3/ item 3
    /conclusion
    
    For each section of the report, do the following:
    - Use simple, clear language
    - Use ## for section title (Markdown format) for each section of the report
    - Do NOT ever refer to yourself as the writer of the report. This should be a professional report without any self-referential language. 
    - Do not say what you are doing in the report. Just write the report without any commentary from yourself.
    - Each section should be as long as necessary to deeply answer the question with the information you have gathered. It is expected that sections will be fairly long and verbose. You are writing a deep research report, and users will expect a thorough answer.
    - Use bullet points to list out information when appropriate, but by default, write in paragraph form.
    
    
    The report will read by user, you need to be careful while generating it.
    Your goal: Deliver excellent, comprehensive, fact-checked reports.

   WORKFLOW EXAMPLES

  **Example 1**

  User Query: "Give me a comprehensive report on Apple's M5 chip"

  Workflow:
  1. ✅ Call web_search_tool(query="Apple M5 chip specifications features", max_results=5, topic="general")
     - Gather technical specifications, release date, features
     - If you decide web_search_tool's response is enough go the the next step, if it isn't call it again.

  2. ✅ Synthesize all information into comprehensive report
     - Structure: Introduction → Technical Specs → Performance → Market Impact → Conclusion
     - Include specific facts, benchmarks, comparisons
     - Add 3-4 most valuable sources in [Title](URL) format

  3. ✅ Call critique_tool(content_to_critique="<your complete report>")
     - Validate report quality, structure, comprehensiveness
     - Check for missing details or weak sections

  4. ✅ Deliver final report to user


  **Example 2: Current Events Research Query**

  User Query: "What's happening with the Ukraine-Russia conflict?"

  Workflow:
  1. ✅ Call guardian_api(query="Ukraine Russia conflict", max_pages=3, page_size=5)
     - Get reliable journalistic coverage
     - Gather recent developments and analysis

  2. ✅ Create well-structured report
     - Structure: Current Situation → Recent Developments → International Response → Analysis → Sources
     - Use clear headings (##) for each section
     - Include specific dates, facts, and key events
     - Reference 3-4 most reliable sources

  3. ✅ Call critique_tool(content_to_critique="<your complete report>")
     - Ensure balanced coverage
     - Verify all key aspects are addressed
     - Check clarity and structure

  6. ✅ Deliver final report to user


  **Key Principles:**
  - Always call critique_tool BEFORE delivering to user
  - Gather information from multiple sources for comprehensive coverage
  - Synthesize information into coherent narrative, not just a list of facts
  - Include only the 3-4 most valuable sources (don't overwhelm with links)
  - Structure reports with clear headings and logical flow

  """

agent = create_deep_agent(
    model=llm,
    tools=[tools_instance.web_search_tool(summarize=True),
           tools_instance.sports_rss_feed(),
           tools_instance.technology_rss_feed(),
           tools_instance.guardian_api_tool(),
           tools_instance.critique_tool()],
    system_prompt=base_agent_instructions,
   # subagents=[sub_critique_agent],
)


def main():
    os.environ['LANGSMITH_API_KEY'] = getattr(settings, 'LANGSMITH_API_KEY', '')
    os.environ['LANGSMITH_TRACING'] = getattr(settings, 'LANGSMITH_TRACING', 'false')

    try:
        response = agent.invoke({
            "messages": [
                {"role": "user", "content": "I want to buy mac mini m4, what are your thoughts, use critique tool ?"}
                #I heard Apple has release m5 chip, I just want generalized information?
                #I want comprehensive information about Ukraine and Russia war
            ]
        })

        # Extract final report if it exists
        if 'files' in response and '/final_report.md' in response['files']:
            report_content = '\n'.join(response['files']['/final_report.md']['content'])
            print("\n=== FINAL REPORT ===\n")
            print(report_content)
        else:
            # Get the last message from the agent
            print(response['messages'][-1].content)

    except Exception as e:
        print(e)
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()

#I don't need to use write tool, probably it would be better
# convert reflection and critique agent into tools


'''
EVALUATE METHOD 


gemini ve claude code ile veri seti olusturup evaluate edeceksin , 5 er 5 er tane (benim domainim bu, bunun icin 5 tane soru ver ve herbiri icin 1 er adet gold answer ver)

for x,y in (question, answer)

    ilk question i agent a vereceksin ve answer verecek bu cevabi gold answer ile birlikte evaluater



open ai, 5 tane soru ve answer uretecek



number of turns u de hesaplayacak



llm as a judge a bir guide veriyorsun ona bakarak puanlama yapiyor.

'''