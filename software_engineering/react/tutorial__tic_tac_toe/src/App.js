
function Square({value}) {
  return <button className="square">{value}</button>;
}


export default function Board() {
  return (
    <>
        <div className="board-row">
            <Square value='1'></Square> 
            <Square value='2'></Square> 
            <Square value='3'></Square> 
        </div>
            <Square value='4'></Square> 
            <Square value='5'></Square> 
            <Square value='6'></Square> 
        <div className="board-row">
        </div>
            <Square value='7'></Square> 
            <Square value='8'></Square> 
            <Square value='9'></Square> 
        <div className="board-row">
        </div>
    </>
  );
}

// import logo from './logo.svg';
// import './App.css';

// function App() {
//   return (
//     <div className="App">
//       <header className="App-header">
//         <img src={logo} className="App-logo" alt="logo" />
//         <p>
//           Edit <code>src/App.js</code> and save to reload.
//         </p>
//         <a
//           className="App-link"
//           href="https://reactjs.org"
//           target="_blank"
//           rel="noopener noreferrer"
//         >
//           Learn React
//         </a>
//       </header>
//     </div>
//   );
// }
//
// export default App;
