// import 'package:flutter/material.dart';
// import 'package:lash_lift_app/start_screen.dart';

// class GradientContainer extends StatelessWidget {
//   const GradientContainer({super.key});

//   @override
//   Widget build(context) {
//     return Container(
//       decoration: BoxDecoration(
//         gradient: LinearGradient(
//           colors: [
//             const Color.fromARGB(255, 198, 186, 186),
//             const Color.fromARGB(255, 212, 191, 191),
//           ],
//           begin: Alignment.topLeft,
//           end: Alignment.bottomRight,
//         ),
//       ),
//       child: const StartScreen(),
//     );
//   }
// }

import 'package:flutter/material.dart';
import 'package:lash_lift_app/start_screen.dart';

class GradientContainer extends StatelessWidget {
  const GradientContainer({super.key});

  @override
  Widget build(context) {
    return Container(
      decoration: const BoxDecoration(
        gradient: LinearGradient(
          colors: [
            Color(0xFFFFF8E1), // Very light cream color
            Color(0xFFFFECB3), // Light cream/beige
            Color(0xFFE6D7C3), // Light brown/beige
          ],
          begin: Alignment.topCenter,
          end: Alignment.bottomCenter,
        ),
      ),
      child: SafeArea(
        child: Padding(
          padding: const EdgeInsets.symmetric(vertical: 20),
          child: StartScreen(),
        ),
      ),
    );
  }
}
