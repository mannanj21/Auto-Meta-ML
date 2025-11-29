export default function LoadingSpinner({ message }) {
  return (
    <div className="flex flex-col items-center justify-center p-12">
      <div className="relative">
        <div className="animate-spin rounded-full h-16 w-16 border-b-4 border-blue-500"></div>
        <div className="absolute top-1/2 left-1/2 transform -translate-x-1/2 -translate-y-1/2">
          <div className="animate-pulse w-6 h-6 bg-blue-500 rounded-full"></div>
        </div>
      </div>
      {message && (
        <p className="mt-4 text-gray-600 font-medium">{message}</p>
      )}
    </div>
  );
}
