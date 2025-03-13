import streamlit as st
from openai import OpenAI
from anthropic import Anthropic
from dotenv import load_dotenv
import os
import re
import requests
import json
from datetime import datetime, timedelta
from dataclasses import dataclass
from typing import Optional, Dict, Any, List

@dataclass
class ModelInfo:
    name: str
    provider: str
    description: str
    input_price: float  # Pris per million tokens
    output_price: float  # Pris per million tokens
    context_window: Optional[int] = None
    best_for: str = ""
    cached_input_price: Optional[float] = None  # Pris for cached input per million tokens
    max_output_tokens: Optional[int] = None
    training_cutoff: Optional[str] = None
    api_name: Optional[str] = None  # Faktisk API-navn som brukes i API-kall

@dataclass
class Message:
    role: str  # "user" eller "assistant"
    content: str
    tokens: Optional[int] = None
    cost: Optional[float] = None  # Kostnad i USD

# Funksjon for å estimere antall tokens basert på tekst
def estimate_tokens(text):
    if not text:
        return 0
    # Enkel estimering: ca 4 tegn per token for engelsk/norsk
    # Dette er en forenkling, faktisk tokenisering er mer kompleks
    return len(text) // 4

# Funksjon for å hente valutakurs fra Frankfurter API med caching
def get_exchange_rate(from_currency="USD", to_currency="NOK"):
    # Sjekk om vi har en cachet valutakurs som er mindre enn 4 dager gammel
    if "exchange_rate_cache" not in st.session_state:
        st.session_state.exchange_rate_cache = {}
    
    cache_key = f"{from_currency}_{to_currency}"
    current_time = datetime.now()
    
    # Hvis vi har en cachet kurs som er mindre enn 4 dager gammel, bruk den
    if (cache_key in st.session_state.exchange_rate_cache and 
        current_time - st.session_state.exchange_rate_cache[cache_key]["timestamp"] < timedelta(days=4)):
        return st.session_state.exchange_rate_cache[cache_key]["rate"]
    
    # Ellers hent ny kurs fra API
    try:
        response = requests.get(f'https://api.frankfurter.app/latest?from={from_currency}&to={to_currency}')
        data = response.json()
        rate = data["rates"][to_currency]
        
        # Lagre i cache
        st.session_state.exchange_rate_cache[cache_key] = {
            "rate": rate,
            "timestamp": current_time
        }
        
        return rate
    except Exception as e:
        st.warning(f"Kunne ikke hente valutakurs: {str(e)}. Bruker standard kurs 10.5 NOK/USD.")
        return 10.5  # Standard kurs hvis API-kallet feiler

# Funksjon for å beregne kostnad basert på tokens og pris
def calculate_cost(tokens, price_per_million):
    return (tokens * price_per_million) / 1_000_000

# Definer tilgjengelige modeller
MODELS = [
    # OpenAI modeller
    ModelInfo(
        name="GPT-3.5 Turbo",
        provider="OpenAI",
        description="Rask og kostnadseffektiv modell for de fleste oppgaver",
        input_price=0.5,
        output_price=1.5,
        context_window=16385,
        best_for="Generelle oppgaver, chat, enkel koding",
        api_name="gpt-3.5-turbo"
    ),
    ModelInfo(
        name="GPT-3.5 Turbo 16K",
        provider="OpenAI",
        description="Som gpt-3.5-turbo, men med større kontekstvindu",
        input_price=1.0,
        output_price=2.0,
        context_window=16385,
        best_for="Lengre samtaler og dokumentanalyse",
        api_name="gpt-3.5-turbo-16k"
    ),
    ModelInfo(
        name="GPT-4o",
        provider="OpenAI",
        description="Høyintelligent modell for komplekse oppgaver",
        input_price=2.50,
        output_price=10.0,
        context_window=128000,
        best_for="Komplekse oppgaver som krever høy presisjon",
        cached_input_price=1.25,
        api_name="gpt-4o"
    ),
    ModelInfo(
        name="GPT-4o mini",
        provider="OpenAI",
        description="Rimelig liten modell for raske, daglige oppgaver",
        input_price=0.15,
        output_price=0.60,
        context_window=128000,
        best_for="Raske, daglige oppgaver",
        cached_input_price=0.075,
        api_name="gpt-4o-mini"
    ),
    
    # Anthropic modeller - oppdatert med nyeste informasjon
    ModelInfo(
        name="Claude 3.7 Sonnet",
        provider="Anthropic",
        description="Vår mest intelligente modell",
        input_price=3.0,
        output_price=15.0,
        context_window=200000,
        best_for="Høyeste nivå av intelligens og kapabilitet med togglebar utvidet tenkning",
        max_output_tokens=64000,
        training_cutoff="November 2024",
        api_name="claude-3-7-sonnet-20250219"
    ),
    ModelInfo(
        name="Claude 3.5 Sonnet (nyeste)",
        provider="Anthropic",
        description="Vår tidligere mest intelligente modell (oppgradert versjon)",
        input_price=3.0,
        output_price=15.0,
        context_window=200000,
        best_for="Høyt nivå av intelligens og kapabilitet",
        max_output_tokens=8192,
        training_cutoff="April 2024",
        api_name="claude-3-5-sonnet-20241022"
    ),
    ModelInfo(
        name="Claude 3.5 Sonnet (tidligere)",
        provider="Anthropic",
        description="Vår tidligere mest intelligente modell (tidligere versjon)",
        input_price=3.0,
        output_price=15.0,
        context_window=200000,
        best_for="Høyt nivå av intelligens og kapabilitet",
        max_output_tokens=8192,
        training_cutoff="April 2024",
        api_name="claude-3-5-sonnet-20240620"
    ),
    ModelInfo(
        name="Claude 3.5 Haiku",
        provider="Anthropic",
        description="Vår raskeste modell",
        input_price=0.80,
        output_price=4.0,
        context_window=200000,
        best_for="Intelligens med lynrask hastighet",
        max_output_tokens=8192,
        training_cutoff="Juli 2024",
        api_name="claude-3-5-haiku-20241022"
    ),
    ModelInfo(
        name="Claude 3 Opus",
        provider="Anthropic",
        description="Kraftig modell for komplekse oppgaver",
        input_price=15.0,
        output_price=75.0,
        context_window=200000,
        best_for="Toppnivå intelligens, flyt og forståelse",
        max_output_tokens=4096,
        training_cutoff="August 2023",
        api_name="claude-3-opus-20240229"
    ),
    ModelInfo(
        name="Claude 3 Haiku",
        provider="Anthropic",
        description="Raskeste og mest kompakte modell for nesten umiddelbar respons",
        input_price=0.25,
        output_price=1.25,
        context_window=200000,
        best_for="Rask og nøyaktig målrettet ytelse",
        max_output_tokens=4096,
        training_cutoff="August 2023",
        api_name="claude-3-haiku-20240307"
    )
]

# Initialiser session state for chat-historikk
if "messages" not in st.session_state:
    st.session_state.messages = []

if "system_prompt" not in st.session_state:
    st.session_state.system_prompt = "Du er en hjelpsom assistent."

if "total_tokens_used" not in st.session_state:
    st.session_state.total_tokens_used = {"input": 0, "output": 0}

if "total_cost" not in st.session_state:
    st.session_state.total_cost = {"input": 0.0, "output": 0.0}

if "current_model" not in st.session_state:
    st.session_state.current_model = MODELS[0].name

# Last inn miljøvariabler
load_dotenv()

# Konfigurer API klienter
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
anthropic_client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

# Hent valutakurs
usd_to_nok_rate = get_exchange_rate("USD", "NOK")

# Sett opp sideoppsett med tre kolonner: venstre (modellvalg), midten (prompt/output), høyre (prisestimering)
st.set_page_config(layout="wide", page_title="LLM Eksperimentering")

# Opprett tre kolonner for hovedlayouten
left_col, main_col, right_col = st.columns([1, 2, 1])

with left_col:
    st.title("🤖 Modeller")
    
    # Vis modeller gruppert etter leverandør
    providers = list(set(model.provider for model in MODELS))
    selected_provider = st.selectbox("Velg Leverandør", providers)
    
    # Filtrer modeller basert på valgt leverandør
    available_models = [model for model in MODELS if model.provider == selected_provider]
    selected_model_name = st.selectbox(
        "Velg Modell",
        [model.name for model in available_models]
    )
    
    # Finn valgt modell
    selected_model = next(model for model in MODELS if model.name == selected_model_name)
    
    # Lagre valgt modell i session state
    if st.session_state.current_model != selected_model.name:
        st.session_state.current_model = selected_model.name
    
    # Valutavalg
    currency = st.radio("Valuta", ["USD", "NOK"], horizontal=True)
    
    # Vis modellinfo
    st.markdown("---")
    st.subheader("Modellinformasjon")
    st.markdown(f"**Beskrivelse:** {selected_model.description}")
    st.markdown(f"**Best for:** {selected_model.best_for}")
    st.markdown("**Priser:**")
    
    # Vis priser i valgt valuta
    if currency == "USD":
        st.markdown(f"- Input: ${selected_model.input_price:.3f} per million tokens")
        if selected_model.cached_input_price:
            st.markdown(f"- Cached Input: ${selected_model.cached_input_price:.3f} per million tokens")
        st.markdown(f"- Output: ${selected_model.output_price:.3f} per million tokens")
    else:  # NOK
        st.markdown(f"- Input: {selected_model.input_price * usd_to_nok_rate:.2f} NOK per million tokens")
        if selected_model.cached_input_price:
            st.markdown(f"- Cached Input: {selected_model.cached_input_price * usd_to_nok_rate:.2f} NOK per million tokens")
        st.markdown(f"- Output: {selected_model.output_price * usd_to_nok_rate:.2f} NOK per million tokens")
    
    if selected_model.context_window:
        st.markdown(f"**Kontekstvindu:** {selected_model.context_window:,} tokens")
    if selected_model.max_output_tokens:
        st.markdown(f"**Maks output:** {selected_model.max_output_tokens:,} tokens")
    if selected_model.training_cutoff:
        st.markdown(f"**Treningsdata cutoff:** {selected_model.training_cutoff}")
    if selected_model.api_name:
        st.markdown(f"**API-navn:** {selected_model.api_name}")
    
    # Vis valutakurs hvis NOK er valgt
    if currency == "NOK":
        st.markdown(f"**Valutakurs:** 1 USD = {usd_to_nok_rate:.2f} NOK")
        if "exchange_rate_cache" in st.session_state and "USD_NOK" in st.session_state.exchange_rate_cache:
            last_updated = st.session_state.exchange_rate_cache["USD_NOK"]["timestamp"]
            st.markdown(f"**Sist oppdatert:** {last_updated.strftime('%Y-%m-%d %H:%M')}")
            days_until_refresh = 4 - (datetime.now() - last_updated).days
            st.markdown(f"**Oppdateres om:** {days_until_refresh} dager")
    
    # Vis API status
    st.markdown("---")
    st.subheader("API Status")
    
    # OpenAI status
    openai_key = os.getenv("OPENAI_API_KEY")
    if openai_key and openai_key != "your-openai-key-here":
        st.success("OpenAI API Nøkkel er konfigurert")
    else:
        st.error("OpenAI API Nøkkel mangler")
    
    # Anthropic status
    anthropic_key = os.getenv("ANTHROPIC_API_KEY")
    if anthropic_key and anthropic_key != "your-anthropic-key-here":
        st.success("Anthropic API Nøkkel er konfigurert")
    else:
        st.error("Anthropic API Nøkkel mangler")

with main_col:
    st.title("🤖 LLM Eksperimentering")
    
    # System prompt øverst
    system_prompt = st.text_area("System prompt:", 
                                value=st.session_state.system_prompt, 
                                height=100)
    st.session_state.system_prompt = system_prompt
    
    # Vis chat-historikk
    st.subheader("Samtale")
    chat_container = st.container(height=400, border=True)
    
    with chat_container:
        for message in st.session_state.messages:
            if message.role == "user":
                st.markdown(f"**Du:**")
                st.markdown(message.content)
            else:
                st.markdown(f"**Assistent:**")
                st.markdown(message.content)
            
            # Vis token-info og kostnad hvis tilgjengelig
            if message.tokens:
                token_info = f"*{message.tokens:,} tokens*"
                if message.cost:
                    if currency == "USD":
                        token_info += f" *- Kostnad: ${message.cost:.6f}*"
                    else:
                        token_info += f" *- Kostnad: {message.cost * usd_to_nok_rate:.6f} NOK*"
                st.caption(token_info)
            
            st.markdown("---")
    
    # Cached input valg
    use_cached_input = st.checkbox("Bruk cached input", value=False, disabled=selected_model.cached_input_price is None)
    
    # Input-felt for ny melding
    user_prompt = st.text_area("Skriv inn din melding:", height=100)
    
    # Send-knapp
    send_col, clear_col = st.columns([3, 1])
    with send_col:
        send_button = st.button("Send", type="primary", use_container_width=True)
    with clear_col:
        if st.button("Tøm samtale", use_container_width=True):
            st.session_state.messages = []
            st.session_state.total_tokens_used = {"input": 0, "output": 0}
            st.session_state.total_cost = {"input": 0.0, "output": 0.0}
            st.rerun()

with right_col:
    st.title("💰 Prisestimering")
    
    # Dynamisk estimering av tokens basert på tekstlengde
    estimated_input_tokens = estimate_tokens(system_prompt) + estimate_tokens(user_prompt)
    if st.session_state.messages:
        # Legg til estimert token-bruk fra tidligere meldinger
        for msg in st.session_state.messages:
            if msg.tokens and msg.role == "user":
                estimated_input_tokens += msg.tokens
    
    # Vis estimert antall input tokens
    st.metric("Estimert input tokens", f"{estimated_input_tokens:,}")
    
    # Beregn estimert input pris
    input_price = selected_model.cached_input_price if use_cached_input and selected_model.cached_input_price else selected_model.input_price
    estimated_input_cost_usd = calculate_cost(estimated_input_tokens, input_price)
    
    # Vis pris i valgt valuta
    if currency == "USD":
        st.metric("Estimert input kostnad", f"${estimated_input_cost_usd:.6f}")
    else:
        estimated_input_cost_nok = estimated_input_cost_usd * usd_to_nok_rate
        st.metric("Estimert input kostnad", f"{estimated_input_cost_nok:.6f} NOK")
    
    # For output, anta at output er ca 2x input som standard, men bruk faktisk output hvis tilgjengelig
    default_output_multiplier = 2
    
    # Sjekk om vi har en nylig output for å gi bedre estimat
    recent_output_tokens = 0
    if st.session_state.messages and len(st.session_state.messages) >= 2:
        last_message = st.session_state.messages[-1]
        if last_message.role == "assistant" and last_message.tokens:
            recent_output_tokens = last_message.tokens
    
    # Bruk nylig output som utgangspunkt hvis tilgjengelig
    estimated_output_tokens = st.number_input(
        "Estimert output tokens", 
        min_value=0, 
        value=recent_output_tokens if recent_output_tokens > 0 else (estimated_input_tokens * default_output_multiplier if estimated_input_tokens > 0 else 500)
    )
    
    # Beregn estimert output pris
    estimated_output_cost_usd = calculate_cost(estimated_output_tokens, selected_model.output_price)
    
    # Vis pris i valgt valuta
    if currency == "USD":
        st.metric("Estimert output kostnad", f"${estimated_output_cost_usd:.6f}")
    else:
        estimated_output_cost_nok = estimated_output_cost_usd * usd_to_nok_rate
        st.metric("Estimert output kostnad", f"{estimated_output_cost_nok:.6f} NOK")
    
    # Total estimert kostnad
    total_estimated_cost_usd = estimated_input_cost_usd + estimated_output_cost_usd
    
    # Vis total pris i valgt valuta
    if currency == "USD":
        st.metric("Total estimert kostnad", f"${total_estimated_cost_usd:.6f}")
    else:
        total_estimated_cost_nok = total_estimated_cost_usd * usd_to_nok_rate
        st.metric("Total estimert kostnad", f"{total_estimated_cost_nok:.6f} NOK")
    
    # Vis pris per token for referanse
    st.markdown("**Pris per token:**")
    if currency == "USD":
        st.markdown(f"- Input: ${input_price/1_000_000:.8f}")
        st.markdown(f"- Output: ${selected_model.output_price/1_000_000:.8f}")
    else:
        st.markdown(f"- Input: {(input_price/1_000_000) * usd_to_nok_rate:.8f} NOK")
        st.markdown(f"- Output: {(selected_model.output_price/1_000_000) * usd_to_nok_rate:.8f} NOK")
    
    # Vis total token-bruk og kostnad
    st.markdown("---")
    st.subheader("📊 Total token-bruk og kostnad")
    total_input_tokens = st.session_state.total_tokens_used["input"]
    total_output_tokens = st.session_state.total_tokens_used["output"]
    
    # Beregn total kostnad
    total_input_cost_usd = st.session_state.total_cost["input"]
    total_output_cost_usd = st.session_state.total_cost["output"]
    total_cost_usd = total_input_cost_usd + total_output_cost_usd
    
    # Vis i valgt valuta
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Total input tokens", f"{total_input_tokens:,}")
        st.metric("Total output tokens", f"{total_output_tokens:,}")
    
    with col2:
        if currency == "USD":
            st.metric("Total input kostnad", f"${total_input_cost_usd:.6f}")
            st.metric("Total output kostnad", f"${total_output_cost_usd:.6f}")
            st.metric("Total kostnad", f"${total_cost_usd:.6f}")
        else:
            total_input_cost_nok = total_input_cost_usd * usd_to_nok_rate
            total_output_cost_nok = total_output_cost_usd * usd_to_nok_rate
            total_cost_nok = total_cost_usd * usd_to_nok_rate
            st.metric("Total input kostnad", f"{total_input_cost_nok:.6f} NOK")
            st.metric("Total output kostnad", f"{total_output_cost_nok:.6f} NOK")
            st.metric("Total kostnad", f"{total_cost_nok:.6f} NOK")

# Håndter generering av svar når Send-knappen trykkes
if send_button and user_prompt:
    # Legg til brukerens melding i historikken
    user_tokens = estimate_tokens(user_prompt)
    user_message = Message(role="user", content=user_prompt, tokens=user_tokens)
    st.session_state.messages.append(user_message)
    
    try:
        with st.spinner("Genererer svar..."):
            # Bygg meldingshistorikk for API-kall
            if selected_model.provider == "OpenAI":
                # OpenAI API kall
                messages = [{"role": "system", "content": system_prompt}]
                
                # Legg til tidligere meldinger
                for msg in st.session_state.messages:
                    if msg.role == "user":
                        messages.append({"role": "user", "content": msg.content})
                    else:
                        messages.append({"role": "assistant", "content": msg.content})
                
                response = openai_client.chat.completions.create(
                    model=selected_model.api_name,
                    messages=messages
                )
                response_text = response.choices[0].message.content
                
                # Hent faktisk token bruk
                prompt_tokens = response.usage.prompt_tokens
                completion_tokens = response.usage.completion_tokens
                
            else:
                # Anthropic API kall
                messages = []
                
                # Legg til tidligere meldinger
                for msg in st.session_state.messages:
                    if msg.role == "user":
                        messages.append({"role": "user", "content": msg.content})
                    else:
                        messages.append({"role": "assistant", "content": msg.content})
                
                response = anthropic_client.messages.create(
                    model=selected_model.api_name,
                    max_tokens=1024,
                    messages=messages,
                    system=system_prompt
                )
                response_text = response.content[0].text
                
                # Hent faktisk token bruk (Anthropic)
                prompt_tokens = response.usage.input_tokens
                completion_tokens = response.usage.output_tokens
            
            # Beregn faktiske kostnader
            input_cost = calculate_cost(prompt_tokens, input_price)
            output_cost = calculate_cost(completion_tokens, selected_model.output_price)
            
            # Oppdater brukerens melding med faktisk token-bruk og kostnad
            if len(st.session_state.messages) > 0:
                last_user_message = st.session_state.messages[-1]
                if last_user_message.role == "user":
                    last_user_message.tokens = user_tokens
                    last_user_message.cost = calculate_cost(user_tokens, input_price)
            
            # Legg til assistentens svar i historikken med kostnad
            assistant_message = Message(
                role="assistant", 
                content=response_text, 
                tokens=completion_tokens,
                cost=output_cost
            )
            st.session_state.messages.append(assistant_message)
            
            # Oppdater total token-bruk og kostnad
            st.session_state.total_tokens_used["input"] += prompt_tokens
            st.session_state.total_tokens_used["output"] += completion_tokens
            st.session_state.total_cost["input"] += input_cost
            st.session_state.total_cost["output"] += output_cost
            
            # Oppdater siden for å vise den nye meldingen
            st.rerun()
            
    except Exception as e:
        with main_col:
            st.error(f"En feil oppstod: {str(e)}") 