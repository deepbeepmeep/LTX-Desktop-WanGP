import React, {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useState,
} from 'react'
import type { ViewType } from '../types/project-model'
import { useProjects } from './ProjectContext'

interface ViewContextType {
  currentView: ViewType
  setCurrentView: (view: ViewType) => void
  openProject: (projectId: string) => void
  goHome: () => void
}

const ViewContext = createContext<ViewContextType | null>(null)

export function ViewProvider({ children }: { children: React.ReactNode }) {
  const {
    activeProject,
    activateProject,
    clearActiveProject,
    setCurrentTab,
  } = useProjects()
  const [currentView, setCurrentView] = useState<ViewType>('home')

  const openProject = useCallback((projectId: string) => {
    activateProject(projectId)
    setCurrentTab('gen-space')
    setCurrentView('project')
  }, [activateProject, setCurrentTab])

  const goHome = useCallback(() => {
    clearActiveProject()
    setCurrentView('home')
  }, [clearActiveProject])

  useEffect(() => {
    if (currentView === 'project' && !activeProject) {
      setCurrentView('home')
    }
  }, [activeProject, currentView])

  return (
    <ViewContext.Provider value={{ currentView, setCurrentView, openProject, goHome }}>
      {children}
    </ViewContext.Provider>
  )
}

export function useView() {
  const context = useContext(ViewContext)
  if (!context) {
    throw new Error('useView must be used within a ViewProvider')
  }
  return context
}
