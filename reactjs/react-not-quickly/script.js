// import React from 'react';
// import ReactDOM from 'react-dom';

let h1 = React.createElement('h1',null,'Hello World');
class HelloWorld extends React.Component {
    render(){
        return React.createElement('div', null, h1, h1);
    }
}
let hElem = React.createElement(HelloWorld, null);
ReactDOM.render(hElem, document.getElementById("root"));