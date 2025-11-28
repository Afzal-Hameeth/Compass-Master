import { StrictMode } from 'react';
import { createRoot } from 'react-dom/client';
import { createBrowserRouter, RouterProvider } from 'react-router-dom';
import './index.css'
import Hero from './pages/Hero';
import DashLayout from './layouts/DashLayout';
import CompassMaster from './pages/compassMaster';
import Error from './pages/Error';
import ResearchAgent from './pages/researchAgent';

const router = createBrowserRouter([
  {
    path: '/',
    element: <Hero />,
    errorElement: <Error />,
  },
  {
    path: '/dashboard',
    element: <DashLayout />,
    errorElement: <Error />,
    children: [
      {
        index: true,
        element: <CompassMaster />,
        errorElement: <Error />,
      },
      {
        path: 'capabilities',
        element: <CompassMaster />,
        errorElement: <Error />,
      },
      {
        path: 'research-agent',
        element: <ResearchAgent />,
        errorElement: <Error />,
      },
    ],
  },
]);

createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <RouterProvider router={router} />
  </StrictMode>,
)
