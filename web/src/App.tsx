import React, { useState, useEffect } from 'react';
import { Chessboard } from 'react-chessboard';
import { Chess } from 'chess.js';
import './App.css';

interface GameState {
  board: string;
  moves: string[];
  status: string;
}

const App: React.FC = () => {
  const [game, setGame] = useState<Chess>(new Chess());
  const [gameId, setGameId] = useState<string>('');
  const [gameState, setGameState] = useState<GameState | null>(null);
  const [ws, setWs] = useState<WebSocket | null>(null);
  const [color, setColor] = useState<'w' | 'b'>('w');

  useEffect(() => {
    // Initialize game
    const initGame = async () => {
      try {
        const response = await fetch(`${process.env.REACT_APP_API_URL}/game/new`, {
          method: 'POST',
        });
        const data = await response.json();
        setGameId(data.board.split(' ')[-1]);
        setGameState(data);
        setGame(new Chess(data.board));
      } catch (error) {
        console.error('Error initializing game:', error);
      }
    };

    initGame();
  }, []);

  useEffect(() => {
    if (gameId) {
      const websocket = new WebSocket(`${process.env.REACT_APP_API_URL.replace('http', 'ws')}/ws/${gameId}`);
      
      websocket.onmessage = (event) => {
        const data = JSON.parse(event.data);
        setGameState(data);
        setGame(new Chess(data.board));
      };

      setWs(websocket);

      return () => {
        websocket.close();
      };
    }
  }, [gameId]);

  const onDrop = async (sourceSquare: string, targetSquare: string) => {
    try {
      const move = game.move({
        from: sourceSquare,
        to: targetSquare,
        promotion: 'q',
      });

      if (move === null) return false;

      if (ws) {
        ws.send(JSON.stringify({
          type: 'move',
          move: `${sourceSquare}${targetSquare}`,
        }));
      }

      return true;
    } catch (error) {
      console.error('Error making move:', error);
      return false;
    }
  };

  const requestAiMove = async () => {
    if (ws) {
      ws.send(JSON.stringify({
        type: 'ai_move',
      }));
    }
  };

  const handleColorChange = (newColor: 'w' | 'b') => {
    setColor(newColor);
  };

  return (
    <div className="App">
      <header className="App-header">
        <h1>AlphaZero Chess</h1>
      </header>
      
      <div className="game-container">
        <div className="board-container">
          <Chessboard
            position={game.fen()}
            onPieceDrop={onDrop}
            boardWidth={600}
            arePiecesDraggable={color === 'w' ? game.turn() === 'w' : game.turn() === 'b'}
          />
        </div>
        
        <div className="controls">
          <div className="color-selector">
            <button
              className={color === 'w' ? 'active' : ''}
              onClick={() => handleColorChange('w')}
            >
              Play as White
            </button>
            <button
              className={color === 'b' ? 'active' : ''}
              onClick={() => handleColorChange('b')}
            >
              Play as Black
            </button>
          </div>
          
          <div className="game-info">
            <p>Status: {gameState?.status}</p>
            <p>Moves: {gameState?.moves.join(', ')}</p>
          </div>
          
          <button
            className="ai-move"
            onClick={requestAiMove}
            disabled={color === 'w' ? game.turn() === 'w' : game.turn() === 'b'}
          >
            Request AI Move
          </button>
        </div>
      </div>
    </div>
  );
};

export default App; 