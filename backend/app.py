from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
import chess
import torch
from models.transformer import ChessTransformer
from mcts.mcts import MCTS
import json
import os
from typing import Dict, List
from pydantic import BaseModel

app = FastAPI()

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Game state management
class GameState(BaseModel):
    board: str  # FEN string
    moves: List[str]
    status: str

class GameManager:
    def __init__(self):
        self.games: Dict[str, chess.Board] = {}
        self.model = self._load_model()
        self.mcts = MCTS(self.model)

    def _load_model(self):
        try:
            checkpoint_files = [f for f in os.listdir('checkpoints') if f.startswith('iteration_')]
            latest_checkpoint = max(checkpoint_files, key=lambda x: int(x.split('_')[1].split('.')[0]))
            model = ChessTransformer()
            checkpoint = torch.load(f'checkpoints/{latest_checkpoint}')
            model.load_state_dict(checkpoint['model_state_dict'])
            model.eval()
            return model
        except:
            print("No checkpoints found. Starting with untrained model.")
            return ChessTransformer()

    def create_game(self, game_id: str) -> GameState:
        board = chess.Board()
        self.games[game_id] = board
        return self._get_game_state(game_id)

    def make_move(self, game_id: str, move_uci: str) -> GameState:
        if game_id not in self.games:
            raise ValueError("Game not found")
        
        board = self.games[game_id]
        move = chess.Move.from_uci(move_uci)
        
        if move not in board.legal_moves:
            raise ValueError("Illegal move")
        
        board.push(move)
        return self._get_game_state(game_id)

    def get_ai_move(self, game_id: str) -> GameState:
        if game_id not in self.games:
            raise ValueError("Game not found")
        
        board = self.games[game_id]
        move = self.mcts.search(board)
        board.push(move)
        return self._get_game_state(game_id)

    def _get_game_state(self, game_id: str) -> GameState:
        board = self.games[game_id]
        return GameState(
            board=board.fen(),
            moves=[move.uci() for move in board.move_stack],
            status=self._get_game_status(board)
        )

    def _get_game_status(self, board: chess.Board) -> str:
        if board.is_checkmate():
            return "checkmate"
        elif board.is_stalemate():
            return "stalemate"
        elif board.is_insufficient_material():
            return "insufficient_material"
        elif board.is_seventyfive_moves():
            return "seventyfive_moves"
        elif board.is_fivefold_repetition():
            return "fivefold_repetition"
        elif board.is_check():
            return "check"
        else:
            return "playing"

game_manager = GameManager()

@app.post("/game/new")
async def new_game() -> GameState:
    game_id = os.urandom(16).hex()
    return game_manager.create_game(game_id)

@app.post("/game/{game_id}/move")
async def make_move(game_id: str, move_uci: str) -> GameState:
    return game_manager.make_move(game_id, move_uci)

@app.post("/game/{game_id}/ai-move")
async def get_ai_move(game_id: str) -> GameState:
    return game_manager.get_ai_move(game_id)

@app.get("/game/{game_id}")
async def get_game_state(game_id: str) -> GameState:
    return game_manager._get_game_state(game_id)

# WebSocket for real-time updates
@app.websocket("/ws/{game_id}")
async def websocket_endpoint(websocket: WebSocket, game_id: str):
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_text()
            message = json.loads(data)
            
            if message["type"] == "move":
                game_state = game_manager.make_move(game_id, message["move"])
            elif message["type"] == "ai_move":
                game_state = game_manager.get_ai_move(game_id)
            else:
                game_state = game_manager._get_game_state(game_id)
            
            await websocket.send_json(game_state.dict())
    except WebSocketDisconnect:
        pass 