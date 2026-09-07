import type { ApiErrorsOf } from './api-client'

export type LocalGenerationError = {
  status: 'default'
  error: {
    code: 'LOCAL_GENERATION_ERROR'
    message: string
  }
}

export type GenerationError = ApiErrorsOf<'generateVideo'> | ApiErrorsOf<'generateImage'> | LocalGenerationError

export function createLocalGenerationError(message: string): LocalGenerationError {
  return {
    status: 'default',
    error: {
      code: 'LOCAL_GENERATION_ERROR',
      message,
    },
  }
}
