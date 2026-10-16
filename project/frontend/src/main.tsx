import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import App from './App'
import { ROOT_ERROR_OPTIONS } from './errorReporting'
import './styles.css'

// React 19 reports every render error to the root as well as to any boundary.  Without
// these callbacks the production default prints the raw thrown value — including the
// payload a component was rendering (FIX-04B / C-AUD-07).
createRoot(document.getElementById('root')!, ROOT_ERROR_OPTIONS).render(
  <StrictMode>
    <App />
  </StrictMode>,
)
