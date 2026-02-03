# TP5/agent/nodes/stubs.py
from TP5.agent.logger import log_event
from TP5.agent.state import AgentState


def stub_reply(state: AgentState) -> AgentState:
    log_event(state.run_id, "node_start", {"node": "stub_reply"})
    state.draft_v1 = "Bonjour, bien reçu. Je reviens vers vous dès que possible avec les informations nécessaires."  # TODO: message minimal (sera remplacé plus tard)
    log_event(state.run_id, "node_end", {"node": "stub_reply", "status": "ok"})
    return state


def stub_ask_clarification(state: AgentState) -> AgentState:
    log_event(state.run_id, "node_start", {"node": "stub_ask_clarification"})
    state.draft_v1 = "J’ai besoin d’une précision avant de répondre : Quelle est l’échéance attendue et le contexte exacte de la demande? "  # TODO: 1-2 questions génériques (sera remplacé plus tard)
    log_event(state.run_id, "node_end", {"node": "stub_ask_clarification", "status": "ok"})
    return state


def stub_escalate(state: AgentState) -> AgentState:
    log_event(state.run_id, "node_start", {"node": "stub_escalate"})
    state.actions.append({
        "type": "handoff_human",
        "summary": "Escalade demandée: email_id={state.email_id}, sujet='{state.subject}'.",   # TODO: résumé court pour escalade
    })
    log_event(state.run_id, "node_end", {"node": "stub_escalate", "status": "ok"})
    return state


def stub_ignore(state: AgentState) -> AgentState:
    log_event(state.run_id, "node_start", {"node": "stub_ignore"})
    state.actions.append({
        "type": "ignore",
        "reason": "Hors périmètre",  # TODO: raison courte (ex: hors périmètre)
    })
    log_event(state.run_id, "node_end", {"node": "stub_ignore", "status": "ok"})
    return state
  