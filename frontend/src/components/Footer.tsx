export function Footer() {
  return (
    <footer className="mt-12 border-t border-white/20 bg-purple-900/40 py-8 backdrop-blur">
      <div className="container mx-auto text-center text-sm text-white/70">
        <p>Built with ❤️ by CrackNCode</p>
        <p>© {new Date().getFullYear()} All rights reserved.</p>
      </div>
    </footer>
  );
}
